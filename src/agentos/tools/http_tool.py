"""HTTP Tool — Let agents call ANY REST API.

Usage:
    from agentos.tools.http_tool import create_api_tool

    # Create a tool that calls any API
    weather_api = create_api_tool(
        name="weather_api",
        description="Get real weather data for a city",
        url="https://api.open-meteo.com/v1/forecast",
        method="GET",
        params_template={"latitude": "{lat}", "longitude": "{lon}", "current_weather": "true"},
    )
"""

from __future__ import annotations
import httpx
import json
from agentos.core.tool import Tool
from agentos.core.types import ToolCall, ToolResult, ToolParam
import time


def create_api_tool(
    name: str,
    description: str,
    url: str,
    method: str = "GET",
    headers: dict | None = None,
    params_template: dict | None = None,
    body_template: dict | None = None,
    response_parser: str | None = None,
    timeout: float = 30.0,
) -> Tool:
    """Create a tool that calls a REST API.

    The agent provides parameter values, and the tool makes the HTTP request.
    """
    # Build parameter list from templates
    params = []
    all_placeholders = set()

    for template in [params_template, body_template]:
        if template:
            for key, val in template.items():
                if isinstance(val, str) and "{" in val:
                    placeholder = val.strip("{}")
                    if placeholder not in all_placeholders:
                        all_placeholders.add(placeholder)
                        params.append(ToolParam(
                            name=placeholder,
                            type="string",
                            description=f"Value for {placeholder}",
                            required=True,
                        ))

    # If no template placeholders, add a generic query param
    if not params:
        params.append(ToolParam(
            name="query",
            type="string",
            description="Query parameter for the API",
            required=True,
        ))

    def api_caller(**kwargs) -> str:
        try:
            # Fill in templates
            final_params = {}
            if params_template:
                for k, v in params_template.items():
                    if isinstance(v, str) and "{" in v:
                        placeholder = v.strip("{}")
                        final_params[k] = kwargs.get(placeholder, v)
                    else:
                        final_params[k] = v

            final_body = None
            if body_template:
                final_body = {}
                for k, v in body_template.items():
                    if isinstance(v, str) and "{" in v:
                        placeholder = v.strip("{}")
                        final_body[k] = kwargs.get(placeholder, v)
                    else:
                        final_body[k] = v

            # Make request
            with httpx.Client(timeout=timeout) as client:
                if method.upper() == "GET":
                    resp = client.get(url, params=final_params, headers=headers or {})
                elif method.upper() == "POST":
                    resp = client.post(url, json=final_body, params=final_params, headers=headers or {})
                else:
                    resp = client.request(method.upper(), url, params=final_params, json=final_body, headers=headers or {})

                resp.raise_for_status()
                data = resp.json()

                # Parse response if parser specified
                if response_parser:
                    for key in response_parser.split("."):
                        if isinstance(data, dict):
                            data = data.get(key, data)

                # Return readable response
                if isinstance(data, dict):
                    return json.dumps(data, indent=2)[:1000]
                elif isinstance(data, list):
                    return json.dumps(data[:5], indent=2)[:1000]
                else:
                    return str(data)[:1000]

        except httpx.HTTPStatusError as e:
            return f"HTTP Error {e.response.status_code}: {e.response.text[:200]}"
        except httpx.ConnectError:
            return f"Connection error: Could not reach {url}"
        except Exception as e:
            return f"Error: {str(e)[:200]}"

    return Tool(fn=api_caller, name=name, description=description)


# ── Pre-built Tools ──

def calculator_tool() -> Tool:
    """Create a calculator tool for evaluating math expressions."""
    def calculator(expression: str) -> str:
        try:
            allowed = set("0123456789+-*/.() ")
            if not all(c in allowed for c in expression):
                return "Error: Only basic math allowed"
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"

    return Tool(fn=calculator, name="calculator", description="Calculate a math expression like '2+2' or '100*0.15'")


def web_search_tool(api_key: str | None = None) -> Tool:
    """Create a web search tool using DuckDuckGo (no API key needed)."""
    def search(query: str) -> str:
        try:
            resp = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1"},
                timeout=10.0,
            )
            data = resp.json()
            results = []
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText']}")
            for r in data.get("RelatedTopics", [])[:3]:
                if isinstance(r, dict) and r.get("Text"):
                    results.append(f"- {r['Text'][:150]}")
            return "\n".join(results) if results else f"No results found for: {query}"
        except Exception as e:
            return f"Search error: {e}"

    return Tool(fn=search, name="web_search", description="Search the web for current information on any topic")


def weather_tool() -> Tool:
    """Create a real weather tool using Open-Meteo (free, no API key)."""
    CITIES = {
        "boston": (42.36, -71.06),
        "new york": (40.71, -74.01),
        "san francisco": (37.77, -122.42),
        "tokyo": (35.68, 139.69),
        "london": (51.51, -0.13),
        "paris": (48.86, 2.35),
        "mumbai": (19.08, 72.88),
        "sydney": (-33.87, 151.21),
        "berlin": (52.52, 13.41),
        "toronto": (43.65, -79.38),
    }

    def get_real_weather(city: str) -> str:
        city_lower = city.lower().strip()
        coords = CITIES.get(city_lower)
        if not coords:
            return f"City '{city}' not in database. Available: {', '.join(CITIES.keys())}"

        try:
            resp = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "current_weather": "true",
                },
                timeout=10.0,
            )
            data = resp.json()
            weather = data.get("current_weather", {})
            temp_c = weather.get("temperature", "N/A")
            temp_f = round(temp_c * 9/5 + 32, 1) if isinstance(temp_c, (int, float)) else "N/A"
            wind = weather.get("windspeed", "N/A")
            code = weather.get("weathercode", 0)

            conditions = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
                55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                80: "Slight rain showers", 81: "Moderate rain showers", 82: "Heavy rain showers",
                95: "Thunderstorm", 96: "Thunderstorm with hail",
            }
            condition = conditions.get(code, f"Code {code}")

            return f"{city.title()}: {temp_f}°F ({temp_c}°C), {condition}, Wind: {wind} km/h"
        except Exception as e:
            return f"Weather error: {e}"

    return Tool(fn=get_real_weather, name="weather", description="Get current real weather data for a city. Supports major cities worldwide.")


def news_tool() -> Tool:
    """Create a news search tool using DuckDuckGo."""
    def search_news(topic: str) -> str:
        try:
            resp = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": f"{topic} news 2026", "format": "json", "no_html": "1"},
                timeout=10.0,
            )
            data = resp.json()
            results = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"][:200])
            for r in data.get("RelatedTopics", [])[:5]:
                if isinstance(r, dict) and r.get("Text"):
                    results.append(f"- {r['Text'][:150]}")
            return "\n".join(results) if results else f"No news found for: {topic}"
        except Exception as e:
            return f"News search error: {e}"

    return Tool(fn=search_news, name="news_search", description="Search for recent news articles on any topic")