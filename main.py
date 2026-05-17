import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *
from medrax.utils.utils import resolve_openai_client_kwargs

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    openai_kwargs=None,
    llm_backend="openai",
    llm_api_key=None,
    llm_kwargs=None,
    log_dir="logs",
    parallel_tool_calls: bool | None = None,
):
    """Initialize the CXR agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.
        llm_backend (str, optional): Backend for LLM ("openai" or "gemini"). Defaults to "openai".
        llm_api_key (str, optional): API key for the selected LLM backend. Defaults to None.
        llm_kwargs (dict, optional): Additional keyword arguments for the LLM backend.
        log_dir (str, optional): Directory to save tool call logs. Defaults to "logs".

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = all_tools.keys() if tools_to_use is None else tools_to_use
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            try:
                tools_dict[tool_name] = all_tools[tool_name]()
            except Exception as exc:
                print(f"Warning: failed to initialize tool {tool_name}: {exc}")

    checkpointer = MemorySaver()
    openai_kwargs = openai_kwargs or {}
    llm_kwargs = llm_kwargs or {}
    if llm_backend == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as exc:
            raise ImportError(
                "Gemini native backend requires `langchain-google-genai`. "
                "Install it with: pip install langchain-google-genai google-generativeai"
            ) from exc
        api_key = (
            llm_api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
        model = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            top_p=top_p,
            google_api_key=api_key,
            **llm_kwargs,
        )
    else:
        resolved_kwargs = resolve_openai_client_kwargs(
            api_key=openai_kwargs.get("api_key"),
            base_url=openai_kwargs.get("base_url"),
        )
        openai_kwargs.update(resolved_kwargs)
        model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir=log_dir,
        system_prompt=prompt,
        checkpointer=checkpointer,
        parallel_tool_calls=parallel_tool_calls,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the CXR agent application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        "ChestXRayClassifierTool",
        "ChestXRaySegmentationTool",
        "ChestXRayReportGeneratorTool",
        "XRayVQATool",
        # "LlavaMedTool",
        # "XRayPhraseGroundingTool",
        # "ChestXRayGeneratorTool",
    ]

    # Collect the ENV variables
    openai_kwargs = resolve_openai_client_kwargs()

    model_dir = os.getenv("MEDRAX_MODEL_DIR", "model-weights")
    temp_dir = os.getenv("MEDRAX_TEMP_DIR", "temp")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", model_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", model_dir)
    os.environ.setdefault("TORCH_HOME", model_dir)

    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir=model_dir,
        temp_dir=temp_dir,
        device="cuda",  # Change this to the device you want to use
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),  # Set OPENAI_MODEL for local servers
        temperature=0.7,
        top_p=0.95,
        openai_kwargs=openai_kwargs
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)
