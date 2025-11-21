from langchain_groq import ChatGroq
import os
from vector_database import retrieve_docs as retrieve_filtered_docs
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
import time
import logging

# Safe import for groq-specific error class
try:
    from groq import BadRequestError
except Exception:
    class BadRequestError(Exception):
        pass

load_dotenv()

# configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Step 1: Setup LLM (DeepSeek R1 via Groq)
# =========================

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY environment variable not set. "
        "Please set it in Streamlit Secrets (GROQ_API_KEY = \"...\")."
    )

# Use an explicit integer for max_tokens to avoid sending null to the API
llm_model = ChatGroq(
    groq_api_key=groq_api_key,
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=1024,  # changed from None to a safe integer
    reasoning_format="parsed",
)

# =========================
# Step 2: Retrieve Docs
# =========================

def retrieve_docs(query, file_name):
    """Wrapper around vector_database.retrieve_docs."""
    return retrieve_filtered_docs(query, file_name)


def get_context(documents, max_chars=None):
    """
    Build a single context string from retrieved documents.
    Optionally truncate to max_chars for safety with long PDFs.
    """
    if not documents:
        return ""

    texts = []
    for doc in documents:
        # Defensive: not all objects may have page_content
        content = getattr(doc, "page_content", None)
        if content:
            texts.append(content)

    context = "\n\n".join(texts)

    if max_chars is not None and len(context) > max_chars:
        context = context[:max_chars]

    return context


# =========================
# Step 3: Answer Question with Follow-Up Support
# =========================

custom_prompt_template = """
Use the pieces of information provided in the context and previous conversation history to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Previous Conversation:
{history}

Question: {question}
Context: {context}
Answer:
"""


def _truncate_context_in_inputs(inputs, factor=0.8, key="context"):
    """If inputs contain a 'context' string, truncate it by factor and return modified inputs."""
    if key in inputs:
        ctx = inputs[key]
        if isinstance(ctx, str) and len(ctx) > 0:
            new_len = max(100, int(len(ctx) * factor))
            inputs = dict(inputs)
            inputs[key] = ctx[:new_len]
    return inputs


def safe_invoke(chain, inputs, retries=3, base_delay=1.0, max_delay=10.0):
    """
    Invoke a LangChain runnable (chain) with retries and exponential backoff.
    On BadRequestError, attempt to truncate the 'context' key before retrying.
    Returns chain.invoke(...) result or raises the final exception.
    """
    attempt = 0
    delay = base_delay
    while True:
        try:
            logger.info("Invoking chain (attempt %d)...", attempt + 1)
            return chain.invoke(inputs)
        except BadRequestError as bre:
            attempt += 1
            logger.warning("BadRequestError on attempt %d: %s", attempt, getattr(bre, "args", bre))
            if attempt > retries:
                logger.error("Exceeded retries for BadRequestError.")
                raise
            # progressively shorten context to try to fit model limits
            inputs = _truncate_context_in_inputs(inputs, factor=0.75)
        except Exception as e:
            attempt += 1
            logger.warning("Exception on attempt %d: %s", attempt, getattr(e, "args", e))
            if attempt > retries:
                logger.error("Exceeded retries for exception.")
                raise
        # backoff
        logger.info("Sleeping for %.1f seconds before retrying...", delay)
        time.sleep(min(delay, max_delay))
        delay = min(delay * 2, max_delay)


def answer_query(documents, model, query, history: str = ""):
    """
    Answer a user query using retrieved documents and chat history.
    `model` will usually be `llm_model`, but is kept as a parameter for flexibility.
    """
    # Limit context length to avoid hitting Groq's context limits
    context = get_context(documents, max_chars=16000)

    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model

    inputs = {
        "question": query,
        "context": context,
        "history": history,
    }

    try:
        response = safe_invoke(chain, inputs)
    except Exception as e:
        logger.error("Failed to get answer from LLM: %s", e)
        return f"LLM request failed: {e}"
    return response


# =========================
# Step 4: Summarization Function
# =========================

def summarize_document(documents):
    """
    Summarize the given legal document(s) concisely while preserving key details.
    """
    # Smaller limit for summary to be extra safe
    context = get_context(documents, max_chars=12000)

    if not context:
        return "No document text could be extracted to summarize."

    summary_prompt = """
    Summarize the given legal document concisely while preserving key legal details.
    Provide a structured summary that highlights the most important points.

    Document:
    {context}

    Summary:
    """

    prompt = ChatPromptTemplate.from_template(summary_prompt)
    chain = prompt | llm_model

    inputs = {"context": context}

    # Use safe_invoke to retry and truncate context on failures
    try:
        return safe_invoke(chain, inputs)
    except BadRequestError:
        return (
            "LLM request failed due to a BadRequest from the Groq API after retries. "
            "Check model parameters, the model name, and your GROQ_API_KEY."
        )
    except Exception as e:
        return f"LLM request failed: {e}"


# =========================
# Step 5: Generate Downloadable Report using ReportLab
# =========================

def generate_report(user_queries, ai_responses):
    """
    Generate a PDF report of the conversation.
    `user_queries` and `ai_responses` should be lists of the same length.
    `ai_responses` can be strings or LangChain message objects (we'll handle both).
    """
    pdf_path = "AI_Lawyer_Report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AI Lawyer Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Below is a record of your conversation with AI Lawyer.")

    y = 700
    max_width = 450  # Maximum width for text before wrapping
    line_height = 15

    for question, answer in zip(user_queries, ai_responses):
        # Handle answer being a LangChain message vs plain string
        if hasattr(answer, "content"):
            answer_text = answer.content
        else:
            answer_text = str(answer)

        # Question
        c.setFont("Helvetica-Bold", 12)
        q_lines = simpleSplit(f"Q: {question}", "Helvetica-Bold", 12, max_width)

        # Answer
        c.setFont("Helvetica", 12)
        a_lines = simpleSplit(f"A: {answer_text}", "Helvetica", 12, max_width)

        # Draw question lines
        for line in q_lines:
            c.drawString(100, y, line)
            y -= line_height

        # Draw answer lines
        for line in a_lines:
            c.drawString(100, y, line)
            y -= line_height

        # Extra space between Q&A pairs
        y -= 20

        # New page if space is low
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750

    c.save()
    return pdf_path