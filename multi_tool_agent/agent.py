# --- Step 0: Setup and Installation ---
# Required installations:
# pip install google-adk litellm google-generativeai pyowm newsapi-python -q

import os
import warnings
import logging
import asyncio
from typing import Optional, Dict, Any
import weave  # Add weave import

# External API Clients
import pyowm
from newsapi import NewsApiClient

# ADK Components
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For potential multi-model use
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse

# Google Generative AI types
from google.genai import types as google_types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


weave.init("Google-Agent2Agent")

# --- Define Model Constants ---
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

# --- Define Tools ---


# Greeting and Farewell Tools
@weave.op()  # Add weave decorator
def say_hello(name: str = "there") -> str:
    """Provides a simple greeting."""
    logging.info(f"Tool 'say_hello' executed with name: {name}")
    return f"Hello, {name}!"


@weave.op()  # Add weave decorator
def say_goodbye() -> str:
    """Provides a simple farewell message."""
    logging.info("Tool 'say_goodbye' executed.")
    return "Goodbye! Have a great day."


# Real time weather tool


@weave.op()  # Add weave decorator
def get_real_weather(city: str, tool_context: ToolContext) -> dict:
    """Retrieves the current real weather report for a specified city using OpenWeatherMap."""
    logging.info(f"Attempting to get real weather for city: '{city}'")

    owm_api_key = os.environ.get("OWM_API_KEY")
    if not owm_api_key or owm_api_key == "YOUR_OWM_API_KEY":
        logging.error("OWM_API_KEY is missing or still set to placeholder.")
        return {
            "status": "error",
            "error_message": "Configuration Error: Weather API key is not set.",
        }
    logging.debug("OWM API Key found.")

    try:
        # Initialize PyOWM
        logging.debug("Initializing PyOWM...")
        owm = pyowm.OWM(owm_api_key)
        mgr = owm.weather_manager()
        logging.debug("PyOWM Manager created.")

        # Get Weather Observation
        logging.info(f"Querying OpenWeatherMap API for city: {city}")
        observation = mgr.weather_at_place(city)
        logging.debug(
            f"OWM Observation object received: {observation}"
        )  # Log the object

        w = observation.weather
        if w is None:
            # This case might be rare if NotFoundError is caught, but good to check
            logging.error(f"OWM observation for '{city}' did not contain weather data.")
            raise ValueError(f"Weather data not available in OWM response for {city}")
        logging.debug(f"OWM Weather object extracted: {w}")  # Log the object

        # Extract Temperature (Kelvin)
        temp_data = w.temperature("kelvin")
        if not temp_data or "temp" not in temp_data:
            logging.error(f"Could not find 'temp' in OWM temperature data: {temp_data}")
            raise ValueError("Temperature data missing 'temp' key in OWM response.")
        temp_k = temp_data["temp"]  # Direct access after check
        logging.debug(f"Temperature (Kelvin): {temp_k}")

        # Read unit preference from state
        preferred_unit = tool_context.state.get(
            "user_preference_temperature_unit", "Celsius"
        )
        logging.info(
            f"Read 'user_preference_temperature_unit' from state: {preferred_unit}"
        )

        # Convert temperature
        if preferred_unit == "Fahrenheit":
            temp_c = temp_k - 273.15
            temp_value = (temp_c * 9 / 5) + 32
            temp_unit = "°F"
        else:  # Default to Celsius
            temp_value = temp_k - 273.15
            temp_unit = "°C"
        logging.debug(f"Converted temperature: {temp_value:.1f}{temp_unit}")

        # Extract other details
        status = w.detailed_status
        humidity = w.humidity
        wind_speed = w.wind().get("speed", "N/A")
        logging.debug(
            f"Weather details: Status='{status}', Humidity={humidity}%, Wind={wind_speed} m/s"
        )

        # Format Report
        report = (
            f"The current weather in {city.capitalize()} is '{status}' "
            f"with a temperature of {temp_value:.1f}{temp_unit}. "
            f"Humidity is {humidity}%, and wind speed is {wind_speed} m/s."
        )

        result = {"status": "success", "report": report}
        logging.info(f"Successfully generated real weather report for {city}.")

        # Update last checked city in state
        tool_context.state["last_city_checked_stateful"] = city.capitalize()
        logging.info(f"Updated state 'last_city_checked_stateful': {city.capitalize()}")

        return result

    # Specific PyOWM Exceptions
    except pyowm.commons.exceptions.NotFoundError:
        error_msg = (
            f"Sorry, I couldn't find weather information for the location '{city}'."
        )
        logging.warning(f"City '{city}' not found by OpenWeatherMap.")
        return {"status": "error", "error_message": error_msg}
    except (
        pyowm.commons.exceptions.APIRequestError,
        pyowm.commons.exceptions.UnauthorizedError,
    ) as api_err:
        # Handles issues like invalid API key, potentially rate limits depending on specific error
        error_msg = f"There was an issue communicating with the weather service (API Error: {api_err}). Please check the API key and try again later."
        logging.error(f"OpenWeatherMap API Error for {city}: {api_err}", exc_info=True)
        return {"status": "error", "error_message": error_msg}

    # Catch other potential errors during processing
    except Exception as e:
        error_msg = (
            f"An unexpected error occurred while processing weather data for {city}."
        )
        # Log the full error detail for debugging
        logging.error(
            f"Unexpected error in get_real_weather for {city}: {e}", exc_info=True
        )
        # Return a user-friendly message, but the logs will have the details
        return {
            "status": "error",
            "error_message": "Sorry, I encountered an internal error while getting the weather.",
        }


# News Fetching Tool using NewsAPI.org
@weave.op()  # Add weave decorator
def get_latest_news(topic: str) -> dict:
    """Fetches the latest 5-10 news headlines for a given topic using NewsAPI."""
    logging.info(f"Tool 'get_latest_news' called for topic: {topic}")

    news_api_key = os.environ.get("NEWS_API_KEY")
    if not news_api_key or news_api_key == "YOUR_NEWS_API_KEY":
        logging.error("NewsAPI Key is missing or placeholder.")
        return {"status": "error", "error_message": "News API key is not configured."}

    try:
        newsapi = NewsApiClient(api_key=news_api_key)

        # Fetch news articles related to the topic, sorted by published date (latest first)
        # Using 'everything' endpoint for broader search on a topic
        articles_data = newsapi.get_everything(
            q=topic, language="en", sort_by="publishedAt", page_size=10
        )  # Get up to 10 articles

        if articles_data["status"] != "ok":
            raise Exception(
                f"NewsAPI returned status: {articles_data.get('code', 'unknown')} - {articles_data.get('message', 'No message')}"
            )

        articles = articles_data["articles"]

        if not articles:
            summary = f"I couldn't find any recent news articles about '{topic}'."
            logging.warning(f"No news articles found for topic: {topic}")
            return {
                "status": "success",
                "news_summary": summary,
            }  # Success, but no articles

        # Format the top 5-10 headlines
        num_headlines = min(len(articles), 10)  # Show up to 10
        headlines_list = []
        for i, article in enumerate(articles[:num_headlines]):
            title = article.get("title", "No Title")
            source = article.get("source", {}).get("name", "Unknown Source")
            # Optional: Add URL - article.get('url', '')
            headlines_list.append(f"{i+1}. {title} ({source})")

        summary = (
            f"Here are the latest {num_headlines} headlines I found about '{topic}':\n"
            + "\n".join(headlines_list)
        )
        result = {"status": "success", "news_summary": summary}
        logging.info(f"Fetched {num_headlines} news headlines for topic: {topic}")
        return result

    except Exception as e:
        error_msg = f"An error occurred while fetching news: {e}"
        logging.error(f"Error in get_latest_news for {topic}: {e}", exc_info=True)
        return {"status": "error", "error_message": error_msg}


# --- Define Callbacks (Model and Tool Guardrails) ---


# Model Guardrail
def block_keyword_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Inspects user input for 'BLOCK', blocks if found."""
    agent_name = callback_context.agent_name
    logging.debug(f"Callback 'block_keyword_guardrail' running for agent: {agent_name}")
    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents):
            if content.role == "user" and content.parts:
                if isinstance(content.parts[0], google_types.Part) and hasattr(
                    content.parts[0], "text"
                ):
                    last_user_message_text = content.parts[0].text or ""
                    break
    keyword_to_block = "BLOCK"
    if keyword_to_block in last_user_message_text.upper():
        logging.warning(f"Keyword '{keyword_to_block}' found. Blocking LLM call.")
        callback_context.state["guardrail_block_keyword_triggered"] = True
        logging.info("Set state 'guardrail_block_keyword_triggered': True")
        return LlmResponse(
            content=google_types.Content(
                role="model",
                parts=[
                    google_types.Part(
                        text="I cannot process this request (blocked keyword)."
                    )
                ],
            )
        )
    else:
        logging.debug(f"Keyword not found. Allowing LLM call for {agent_name}.")
        return None


# Tool Guardrail
def block_paris_tool_guardrail(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """Blocks 'get_real_weather' tool execution for 'Paris'."""
    tool_name = tool.name
    agent_name = tool_context.agent_name
    logging.debug(
        f"Callback 'block_paris_tool_guardrail' running for tool '{tool_name}' in agent '{agent_name}'"
    )
    logging.debug(f"Inspecting tool args: {args}")

    # *** UPDATED target_tool_name ***
    target_tool_name = "get_real_weather"
    blocked_city = "paris"

    if tool_name == target_tool_name:
        city_argument = args.get("city", "")
        if city_argument and city_argument.lower() == blocked_city:
            logging.warning(
                f"Blocked city '{city_argument}' detected for tool '{tool_name}'. Blocking execution."
            )
            tool_context.state["guardrail_tool_block_triggered"] = True
            logging.info("Set state 'guardrail_tool_block_triggered': True")
            return {  # Return error dictionary, skipping the actual tool
                "status": "error",
                "error_message": f"Policy restriction: Weather checks for '{city_argument.capitalize()}' are disabled by a tool guardrail.",
            }
        else:
            logging.debug(f"City '{city_argument}' is allowed for tool '{tool_name}'.")
    else:
        logging.debug(f"Tool '{tool_name}' not targeted by Paris guardrail. Allowing.")

    logging.debug(f"Allowing tool '{tool_name}' to proceed.")
    return None  # Allow tool execution


# --- Define Agents (Sub-Agents, News Agent, and Final Root Agent) ---


# Greeting and Farewell Agents
greeting_agent = None
farewell_agent = None
try:
    greeting_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="greeting_agent",
        instruction="Greet the user friendly.",
        description="Handles simple greetings and hellos.",
        tools=[say_hello],
    )
    logging.info(f"Agent '{greeting_agent.name}' created successfully.")
except Exception as e:
    logging.error(f"Failed to create Greeting agent: {e}", exc_info=True)

try:
    farewell_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,
        name="farewell_agent",
        instruction="Provide a polite goodbye.",
        description="Handles simple farewells and goodbyes.",
        tools=[say_goodbye],
    )
    logging.info(f"Agent '{farewell_agent.name}' created successfully.")
except Exception as e:
    logging.error(f"Failed to create Farewell agent: {e}", exc_info=True)


# News Agent Definition
news_agent = None
try:
    news_agent = Agent(
        model=MODEL_GEMINI_2_0_FLASH,  # Can use a different model if desired
        name="news_agent",
        instruction="You are a News Reporter agent. Your goal is to fetch and present the latest news headlines on a specific topic requested by the user. Use the 'get_latest_news' tool. Clearly state the topic and present the headlines returned by the tool. If the tool returns an error or no news, inform the user politely.",
        description="Fetches and presents the latest 5-10 news headlines for a given topic using the 'get_latest_news' tool.",
        tools=[get_latest_news],
    )
    logging.info(f"Agent '{news_agent.name}' created successfully.")
except Exception as e:
    logging.error(f"Failed to create News agent: {e}", exc_info=True)


# Define Final Root Agent
root_agent = None  # Renamed for ADK CLI compatibility
if greeting_agent and farewell_agent and news_agent:  # Check all required sub-agents
    try:
        # *** UPDATED instruction, tools, and sub_agents ***
        root_agent = Agent(
            name="weather_news_assistant",
            model=MODEL_GEMINI_2_0_FLASH,  # Orchestration model
            description="Main assistant: Handles real weather requests, delegates news requests, greetings, and farewells. Includes safety guardrails.",
            instruction=(
                "You are the main Assistant coordinating a team. Your primary responsibilities are providing real-time weather and delegating other tasks.\n"
                "1.  **Weather:** If the user asks for weather in a specific city, use the 'get_real_weather' tool yourself. The tool respects temperature unit preferences stored in state.\n"
                "2.  **News:** If the user asks for news on a specific topic (e.g., 'latest news on AI', 'updates on electric vehicles'), delegate the request to the 'news_agent'.\n"
                "3.  **Greetings:** If the user offers a simple greeting ('Hi', 'Hello'), delegate to the 'greeting_agent'.\n"
                "4.  **Farewells:** If the user says goodbye ('Bye', 'Thanks bye'), delegate to the 'farewell_agent'.\n"
                "Analyze the user's query and delegate or handle it appropriately. If unsure, ask for clarification. Only use tools or delegate as described."
            ),
            tools=[get_real_weather],  # Root agent handles weather directly
            sub_agents=[greeting_agent, farewell_agent, news_agent],
            output_key="last_assistant_response",
            before_model_callback=block_keyword_guardrail,
            before_tool_callback=block_paris_tool_guardrail,
        )
        logging.info(f"Root Agent '{root_agent.name}' created successfully.")
    except Exception as e:
        logging.error(f"Failed to create Root agent: {e}", exc_info=True)
        root_agent = None  # Ensure it's None if creation failed
else:
    logging.warning(
        "Skipping Root agent definition because one or more sub-agents failed."
    )


# --- Setup Session Service and Runner ---
APP_NAME = "weather_news_app"
USER_ID_STATEFUL = "user_weather_news_demo"
SESSION_ID_STATEFUL = "session_weather_news_001"

session_service_stateful = InMemorySessionService()
logging.info("InMemorySessionService created.")

initial_state = {"user_preference_temperature_unit": "Celsius"}
session_stateful = None
try:
    session_stateful = session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID_STATEFUL,
        session_id=SESSION_ID_STATEFUL,
        state=initial_state,
    )
    print(
        f"✅ Session '{SESSION_ID_STATEFUL}' created with initial state: {initial_state}"
    )
    logging.info(
        f"Session '{SESSION_ID_STATEFUL}' created with initial state: {initial_state}"
    )
except Exception as e:
    print(f"❌ Error creating session: {e}")
    logging.error(f"Failed to create session: {e}", exc_info=True)


runner_root = None
if root_agent and session_stateful:
    try:
        runner_root = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service_stateful,
        )
        print(f"✅ Runner created for final root agent '{runner_root.agent.name}'.")
        logging.info(f"Runner created for agent '{runner_root.agent.name}'.")
    except Exception as e:
        print(f"❌ Error creating runner: {e}")
        logging.error(f"Failed to create runner: {e}", exc_info=True)
else:
    print("❌ Skipping Runner creation because Root Agent or Session is missing.")
    logging.warning(
        "Skipping Runner creation because Root Agent or Session is missing."
    )


# --- Define Agent Interaction Function ---
async def call_agent_async(
    query: str, runner: Optional[Runner], user_id: str, session_id: str
):
    """Sends query to agent via runner, prints final response."""
    print(f"\n>>> User Query: {query}")
    logging.info(f"User Query: {query}")
    if not runner:
        print("<<< Agent Response: Error - Runner not available.")
        logging.error("call_agent_async called but runner is None.")
        return
    content = google_types.Content(role="user", parts=[google_types.Part(text=query)])
    final_response_text = "Agent did not produce a final response."
    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=content
        ):
            logging.debug(
                f"Event: Author={event.author}, Type={type(event).__name__}, Final={event.is_final_response()}"
            )
            if event.is_final_response():
                logging.debug("Final response event identified.")
                if event.content and event.content.parts:
                    if isinstance(
                        event.content.parts[0], google_types.Part
                    ) and hasattr(event.content.parts[0], "text"):
                        final_response_text = event.content.parts[0].text or ""
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    logging.warning(f"Agent escalated: {event.error_message}")
                break
    except Exception as e:
        final_response_text = f"Error during agent execution: {e}"
        print(f"!!! Exception during run_async: {e}")
        logging.error(f"Exception during run_async: {e}", exc_info=True)
    print(f"<<< Agent Response: {final_response_text}")
    logging.info(f"Agent Response: {final_response_text}")


@weave.op(call_display_name="Main Conversation")
async def main_conversation():
    print("\n--- Starting Standalone Test Conversation (Weather & News) ---")
    logging.info("Starting standalone test conversation (Weather & News).")
    if not runner_root:
        print("❌ Cannot run conversation: Final Runner was not created.")
        logging.critical("Cannot run test conversation: Runner is None.")
        return

    async def interact(query):
        await call_agent_async(
            query, runner_root, USER_ID_STATEFUL, SESSION_ID_STATEFUL
        )

    # Test Interactions:
    print("\n--- Turn 1: Greeting (Delegation) ---")
    await interact("Hello")

    print("\n--- Turn 2: Real Weather (Root Agent Tool, Expect Celsius) ---")
    await interact("What's the weather in California?")

    print(
        "\n--- Interlude: Manually Updating Temp Pref to Fahrenheit (Testing State) ---"
    )
    try:
        stored_session = session_service_stateful.sessions[APP_NAME][USER_ID_STATEFUL][
            SESSION_ID_STATEFUL
        ]
        stored_session.state["user_preference_temperature_unit"] = "Fahrenheit"
        print(
            f"--- Internal session state updated. Temp Pref: {stored_session.state['user_preference_temperature_unit']} ---"
        )
        logging.info(
            "Manually updated state 'user_preference_temperature_unit' to Fahrenheit for testing."
        )
    except Exception as e:
        print(f"--- Error updating internal session state: {e} ---")
        logging.error(f"Failed to manually update state: {e}", exc_info=True)

    print("\n--- Turn 3: Real Weather Again (Expect Fahrenheit) ---")
    await interact("How's the weather in London?")

    print("\n--- Turn 4: Question on the weather in London ---")
    await interact("Should I go out without an umbrella if I am currently in London?")

    print("\n--- Turn 5: News Request (Delegation to News Agent) ---")
    await interact("Tell me the latest news about AI")

    print("\n--- Turn 6: Test Input Guardrail (Blocked Keyword) ---")
    await interact("BLOCK this request")

    print("\n--- Turn 7: Test Tool Guardrail (Paris Weather) ---")
    await interact("What is the weather in Paris?")

    print("\n--- Turn 8: Farewell (Delegation) ---")
    await interact("Okay that's all, thanks bye!")

    # Inspect Final Session State
    print("\n--- Inspecting Final Session State ---")
    logging.info("Inspecting final session state.")
    final_session = session_service_stateful.get_session(
        app_name=APP_NAME, user_id=USER_ID_STATEFUL, session_id=SESSION_ID_STATEFUL
    )
    if final_session:
        print(
            f"Final Temp Preference: {final_session.state.get('user_preference_temperature_unit')}"
        )
        print(
            f"Final Last Assistant Response (output_key): {final_session.state.get('last_assistant_response')}"
        )
        print(
            f"Final Last City Checked (weather tool): {final_session.state.get('last_city_checked_stateful')}"
        )
        print(
            f"Model Guardrail Triggered Flag: {final_session.state.get('guardrail_block_keyword_triggered')}"
        )
        print(
            f"Tool Guardrail Triggered Flag: {final_session.state.get('guardrail_tool_block_triggered')}"
        )
        logging.info(f"Final session state: {final_session.state}")
    else:
        print("\n❌ Error: Could not retrieve final session state.")
        logging.error("Could not retrieve final session state.")

    print("\n--- Standalone Test Conversation Complete ---")
    logging.info("Standalone test conversation complete.")


# --- Script Entry Point ---
if __name__ == "__main__":
    if "root_agent" in globals() and root_agent is not None:
        print("Root agent found. Running test conversation...")
        try:
            asyncio.run(main_conversation())
        except Exception as e:
            print(f"\n💥 An error occurred during the test conversation: {e}")
            logging.critical(
                f"An error occurred during the test conversation: {e}", exc_info=True
            )
    else:
        print(
            "\n❌ Cannot run test conversation: 'root_agent' was not defined successfully."
        )
        logging.critical(
            "Cannot run test conversation: 'root_agent' is None or not defined."
        )

    print("\n--- Script Execution Finished ---")
