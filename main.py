
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple, List, Any
from urllib.parse import urljoin, urlparse

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 300
MAX_RETRIES = 3
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.0-flash"

ALLOWED_SCHEMES = {"https", "http"}

http_client: Optional[httpx.AsyncClient] = None
gemini_client: Optional[OpenAI] = None

def create_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(TIMEOUT_SECONDS),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=50),
        follow_redirects=True,
        verify=True,
    )

def create_gemini_client() -> Optional[OpenAI]:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        logger.warning("GEMINI_API_KEY not found. Message condensing will be disabled.")
        return None
    
    try:
        client = OpenAI(
            api_key=gemini_api_key,
            base_url=GEMINI_BASE_URL
        )
        
        test_response = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10
        )
        
        if test_response and test_response.choices:
            logger.info("Gemini client initialized and tested successfully")
            return client
        else:
            logger.error("Gemini test request failed - no response")
            return None
            
    except Exception as e:
        logger.error(f"Failed to initialize or test Gemini client: {e}")
        return None

def parse_proxy_url(path: str) -> Tuple[str, str]:
    if not path or path == "/":
        raise HTTPException(
            status_code=400, 
            detail="URL format required: /{target_domain}/{path}"
        )
    
    clean_path = path.lstrip("/")
    if not clean_path:
        raise HTTPException(
            status_code=400, 
            detail="URL format required: /{target_domain}/{path}"
        )
    
    path_parts = clean_path.split("/")
    if len(path_parts) < 1:
        raise HTTPException(
            status_code=400, 
            detail="URL format required: /{target_domain}/{path}"
        )
    
    domain = path_parts[0]
    target_path = "/" + "/".join(path_parts[1:]) if len(path_parts) > 1 else "/"
    
    if not domain or "." not in domain and ":" not in domain:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid domain '{domain}'"
        )
    
    if domain.startswith("localhost") or (":" in domain and not domain.startswith("localhost")):
        protocol = "http"
    else:
        protocol = "https"
    
    target_base_url = f"{protocol}://{domain}"
    
    try:
        parsed = urlparse(target_base_url)
        if parsed.scheme not in ALLOWED_SCHEMES or not parsed.netloc:
            raise ValueError("Invalid URL structure")
    except Exception:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid target URL: {target_base_url}"
        )
    
    logger.info(f"Parsed: {path} → {target_base_url}{target_path}")
    return target_base_url, target_path

async def condense_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not gemini_client:
        logger.warning("Gemini client not available. Skipping message condensing.")
        return messages
    
    if not messages or len(messages) < 2:
        return messages
    
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    conversational_messages = [msg for msg in messages if msg.get("role") in ["user", "assistant"]]
    
    if len(conversational_messages) < 2:
        logger.debug("Fewer than 2 conversational messages. Skipping condensing.")
        return messages
    
    logger.info(f"Preserving {len(system_messages)} system messages unchanged")
    logger.info(f"Condensing {len(conversational_messages)} user/assistant messages")
    
    try:
        condensing_prompt = f"""You are an expert at condensing messages while preserving important information.

Task: Make these messages shorter while keeping ALL key details, code, URLs, and technical info exactly as written.

Input messages:
{json.dumps(conversational_messages, indent=2)}

Return ONLY this JSON format:
{{"condensed_messages": [
  {{"role": "user", "content": "shortened version"}},
  {{"role": "assistant", "content": "shortened version"}}
]}}

Rules:
- Keep code blocks, URLs, commands exactly as written
- Preserve technical details and numbers
- Remove only unnecessary words
- Maintain the same roles and message count"""

        response = gemini_client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "user", "content": condensing_prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        if not response or not response.choices or not response.choices[0].message:
            logger.error("Invalid response structure from Gemini")
            return messages
            
        condensed_response = response.choices[0].message.content
        
        if not condensed_response:
            logger.error("Empty response content from Gemini")
            return messages
            
        condensed_response = condensed_response.strip()
        logger.debug(f"Raw Gemini response (first 200 chars): {condensed_response[:200]}...")
        
        if condensed_response.startswith("```json"):
            condensed_response = condensed_response[7:]
        if condensed_response.startswith("```"):
            condensed_response = condensed_response[3:]
        if condensed_response.endswith("```"):
            condensed_response = condensed_response[:-3]
        
        condensed_response = condensed_response.strip()
        
        try:
            parsed_response = json.loads(condensed_response)
            condensed_conversational = parsed_response.get("condensed_messages", [])
            
            if not condensed_conversational:
                logger.warning("Gemini returned empty condensed messages. Using original.")
                return messages
            
            for msg in condensed_conversational:
                if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    logger.warning("Invalid condensed message structure. Using original.")
                    return messages
            
            original_length = sum(len(str(msg.get("content", ""))) for msg in conversational_messages)
            condensed_length = sum(len(str(msg.get("content", ""))) for msg in condensed_conversational)
            
            if condensed_length > 0:
                compression_ratio = original_length / condensed_length
                logger.info(f"AI Condensing SUCCESS: {original_length} → {condensed_length} chars (ratio: {compression_ratio:.1f}x)")
                logger.info(f"Estimated token savings: {((compression_ratio - 1) / compression_ratio * 100):.0f}%")
            else:
                logger.warning("AI Condensing resulted in empty messages. Using original.")
                return messages
            
            result_messages = []
            conversational_index = 0
            
            for original_msg in messages:
                if original_msg.get("role") == "system":
                    result_messages.append(original_msg)
                elif original_msg.get("role") in ["user", "assistant"]:
                    if conversational_index < len(condensed_conversational):
                        result_messages.append(condensed_conversational[conversational_index])
                        conversational_index += 1
                    else:
                        result_messages.append(original_msg)
                else:
                    result_messages.append(original_msg)
            
            logger.info(f"Final result: {len(system_messages)} system + {len(condensed_conversational)} condensed messages")
            return result_messages
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini condensing response as JSON: {e}")
            logger.error(f"Full Gemini response: {condensed_response}")
            return messages
            
    except Exception as e:
        logger.error(f"Error during message condensing: {e}")
        return messages

async def process_request_body(request: Request, content: bytes) -> bytes:
    if not content or request.method not in ["POST", "PUT", "PATCH"]:
        return content
    
    try:
        body = json.loads(content.decode('utf-8'))
        
        if isinstance(body, dict) and "messages" in body:
            messages = body.get("messages", [])
            
            if isinstance(messages, list) and len(messages) > 1:
                logger.info(f"Chat completion detected: {len(messages)} messages found")
                logger.info(f"Starting AI condensing with Gemini 2.0 Flash...")
                
                condensed_messages = await condense_messages(messages)
                
                body["messages"] = condensed_messages
                
                logger.info(f"Request body updated with condensed messages")
                return json.dumps(body).encode('utf-8')
            else:
                logger.debug("Chat completion with ≤1 message. Skipping condensing.")
        else:
            logger.debug("Non-chat request: Passing through unchanged (no 'messages' field)")
        
        return content
        
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.debug(f"Could not parse request body as JSON: {e}")
        return content
    except Exception as e:
        logger.error(f"Error processing request body: {e}")
        return content

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, gemini_client
    
    http_client = create_http_client()
    gemini_client = create_gemini_client()
    
    logger.info("Universal API Proxy with AI Message Condensing started successfully")
    if gemini_client:
        logger.info("AI message condensing enabled (Gemini 2.0 Flash)")
    else:
        logger.warning("AI message condensing disabled (no GEMINI_API_KEY)")
    
    yield
    
    if http_client:
        await http_client.aclose()
    logger.info("Universal API Proxy shut down")

app = FastAPI(
    title="Universal API Proxy with AI Condensing",
    description="Direct URL mapping with intelligent message compression using Gemini 2.0 Flash",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_filtered_headers(request: Request) -> Dict[str, str]:
    excluded_headers = {
        'host', 'content-length', 'connection', 'keep-alive',
        'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 
        'transfer-encoding', 'upgrade'
    }
    
    headers = {}
    for name, value in request.headers.items():
        if name.lower() not in excluded_headers:
            headers[name] = value
    
    if request.method in ["POST", "PUT", "PATCH"] and "content-type" not in headers:
        headers["content-type"] = "application/json"
    
    return headers


@app.get("/")
async def root():
    return {
        "service": "Universal API Proxy",
        "version": "5.0.0",
        "format": "/{target_domain}/{path}",
        "ai_condensing": "enabled" if gemini_client else "disabled"
    }

async def forward_request(
    method: str,
    base_url: str,
    path: str,
    headers: Dict[str, str],
    content: bytes = b"",
    params: Optional[Dict[str, str]] = None
) -> httpx.Response:
    if not http_client:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")
    
    target_url = urljoin(base_url, path.lstrip('/'))
    logger.info(f"Forwarding {method} request to: {target_url}")
    
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await http_client.request(
                method=method,
                url=target_url,
                headers=headers,
                content=content,
                params=params
            )
            return response
            
        except httpx.TimeoutException as e:
            last_exception = e
            logger.warning(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES} for {target_url}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                
        except httpx.NetworkError as e:
            last_exception = e
            logger.warning(f"Network error on attempt {attempt + 1}/{MAX_RETRIES}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)
                
        except Exception as e:
            logger.error(f"Unexpected error during request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
    
    error_msg = f"Failed to connect to target API after {MAX_RETRIES} attempts"
    if last_exception:
        error_msg += f": {str(last_exception)}"
    
    logger.error(error_msg)
    raise HTTPException(status_code=502, detail=error_msg)

async def stream_response(response: httpx.Response):
    try:
        async for chunk in response.aiter_bytes(chunk_size=8192):
            yield chunk
    except Exception as e:
        logger.error(f"Error streaming response: {str(e)}")
        yield b""

def create_response_headers(upstream_response: httpx.Response) -> Dict[str, str]:
    excluded_headers = {
        'content-length', 'transfer-encoding', 'connection', 
        'keep-alive', 'upgrade', 'proxy-authenticate', 'proxy-authorization'
    }
    
    headers = {}
    for name, value in upstream_response.headers.items():
        if name.lower() not in excluded_headers:
            headers[name] = value
    
    return headers

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_request(request: Request, path: str):
    
    if path == "":
        raise HTTPException(
            status_code=404,
            detail="Endpoint not found in proxy route"
        )
    
    try:
        method = request.method
        headers = get_filtered_headers(request)
        params = dict(request.query_params) if request.query_params else None
        
        target_base_url, target_path = parse_proxy_url(path)
        
        raw_content = b""
        if method in ["POST", "PUT", "PATCH"]:
            raw_content = await request.body()
            
        processed_content = await process_request_body(request, raw_content)
        
        if processed_content != raw_content and processed_content:
            headers["content-length"] = str(len(processed_content))
        
        upstream_response = await forward_request(
            method=method,
            base_url=target_base_url,
            path=target_path,
            headers=headers,
            content=processed_content,
            params=params
        )
        
        response_headers = create_response_headers(upstream_response)
        
        if upstream_response.headers.get("content-type", "").startswith("text/event-stream"):
            return StreamingResponse(
                stream_response(upstream_response),
                status_code=upstream_response.status_code,
                headers=response_headers,
                media_type="text/event-stream"
            )
        
        content = await upstream_response.aread()
        
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=response_headers
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error in proxy_request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}")

@app.api_route("/", methods=["POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"], include_in_schema=False)
async def root_other_methods():
    raise HTTPException(
        status_code=400,
        detail="URL format required: /{target_domain}/{path}"
    )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0", 
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )
