from openai import OpenAI
import time
import json
import re
from openai import APITimeoutError, APIConnectionError, RateLimitError

#deepseek/deepseek-chat-v3-0324:free,deepseek/deepseek-r1-0528,
class LLMClient:
    def __init__(self, api_key, model="openai/gpt-4o-mini"):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/forex-trading-bot", # Required by OpenRouter
                "X-Title": "UFO Forex Trading Bot", # Optional but recommended
            }
        )
        self.model = model
        self.max_retries = 5  # Increased retries for better reliability
        self.retry_delay = 3  # Longer delay to avoid rate limits

    def generate_response(self, prompt, timeout=60):
        """
        Generates a response from the LLM with robust error handling and retries.
        """
        for attempt in range(self.max_retries):
            try:
                print(f"LLM Request (attempt {attempt + 1}/{self.max_retries})...")
                
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,  # Ensure we get complete responses
                    temperature=0.7,  # Balanced creativity for trading decisions
                    timeout=timeout,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/forex-trading-bot",
                        "X-Title": "UFO Forex Trading Bot"
                    }
                )
                
                response_content = completion.choices[0].message.content
                
                # Validate JSON if response contains JSON
                if '{' in response_content and '}' in response_content:
                    self._validate_json_response(response_content)
                
                print(f"LLM Response received successfully")
                return response_content
                
            except json.JSONDecodeError as e:
                print(f"JSON parse error in LLM response: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    print("Max retries exceeded for JSON parsing")
                    return self._generate_fallback_response()
                    
            except RateLimitError as e:
                print(f"Rate limit hit calling OpenRouter LLM: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff for rate limits
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited - waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    print("Max retries exceeded for rate limits")
                    return self._generate_fallback_response()
                    
            except (APITimeoutError, APIConnectionError) as e:
                print(f"Network error calling OpenRouter LLM: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    print("Max retries exceeded for network errors")
                    return self._generate_fallback_response()
                    
            except Exception as e:
                print(f"Error calling OpenRouter LLM: {e}")
                if attempt < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    continue
                else:
                    print("Max retries exceeded for unexpected errors")
                    return self._generate_fallback_response()
        
        # If we get here, all retries failed
        return self._generate_fallback_response()
    
    def _validate_json_response(self, response_content):
        """Validate JSON content in the response"""
        try:
            # Only validate if response looks like it contains JSON
            if '{' in response_content and '}' in response_content:
                # Try to extract and parse JSON from the response
                json_match = re.search(r'{.*}', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    # Clean up common JSON issues
                    json_str = re.sub(r'//.*?\n', '\n', json_str)  # Remove comments
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)  # Remove trailing commas
                    
                    # Try to parse - if it fails, it's not critical for non-JSON responses
                    try:
                        json.loads(json_str)
                    except json.JSONDecodeError:
                        # For non-JSON responses (like analysis text), this is acceptable
                        if not any(keyword in response_content.lower() for keyword in ['trades', 'actions', 'currency_pair']):
                            return  # Skip validation for text-only responses
                        # Only raise error for responses that should contain valid JSON
                        raise
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Invalid JSON in LLM response", response_content, 0)
    
    def _generate_fallback_response(self):
        """Generate a fallback response when LLM fails"""
        return '''{
    "analysis": "LLM service temporarily unavailable - using conservative approach",
    "consensus": "Hold existing positions due to technical issues with market analysis",
    "trades": [],
    "recommendation": "Wait for service restoration before making new trading decisions"
}'''
