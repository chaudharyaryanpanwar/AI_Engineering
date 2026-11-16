# ============================================================
# LANGCHAIN ASSIGNMENT
# Q1: E-commerce Customer Support Assistant
# Q2: RAG-based Customer Support Assistant
#
# Before running:
# 1. Install the required packages.
# 2. Set your AWS credentials as environment variables.
# 3. Place the Excel file at:
#       Dataset/ecommerce_customer_support.xlsx
#
# Install packages:
# pip install -U langchain langchain-aws langchain-community \
#     langchain-classic langchain-text-splitters faiss-cpu \
#     pandas openpyxl
#
# Set AWS credentials (Mac/Linux):
# export AWS_ACCESS_KEY_ID="your_access_key"
# export AWS_SECRET_ACCESS_KEY="your_secret_key"
#
# Set AWS credentials (Windows PowerShell):
# $env:AWS_ACCESS_KEY_ID="your_access_key"
# $env:AWS_SECRET_ACCESS_KEY="your_secret_key"
# ============================================================


import os
import pandas as pd

from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ------------------------------------------------------------
# COMMON: LOAD DATASET
# ------------------------------------------------------------

DATASET_PATH = "Dataset/ecommerce_customer_support.xlsx"

df = pd.read_excel(DATASET_PATH)

print("Dataset loaded successfully.")
print(df.head())


# ============================================================
# Q1: CUSTOMER SUPPORT ASSISTANT WITH CHAT HISTORY
# ============================================================

print("\n" + "=" * 60)
print("Q1: CUSTOMER SUPPORT ASSISTANT")
print("=" * 60)


# Step 1: Create the Amazon Bedrock chat model

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0.2,
        "max_tokens": 500
    }
)


# Step 2: Get one example from the dataset for one-shot prompting

example = df.iloc[0]

example_query = str(example["customer_query"])
example_response = str(example["expected_response"])


# Step 3: Create the prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are ShopEase AI Support Assistant.

            Your job is to help customers with:
            - Orders
            - Returns
            - Shipping
            - Product information

            Instructions:
            1. Be polite, helpful, and empathetic.
            2. Identify the customer's main issue.
            3. Give a clear and useful answer.
            4. Keep the answer within 150 words.
            5. If the issue is not clear, ask a short follow-up question.

            Use this example to understand the expected response style.

            Example customer query:
            {example_query}

            Example expected response:
            {example_response}
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{customer_query}")
    ]
)


# Step 4: Build the LCEL chain

chain = prompt | llm | StrOutputParser()


# Step 5: Create session-based chat history

store = {}


def get_session_history(session_id: str):
    """Return the chat history for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# Step 6: Add memory to the chain

assistant = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="customer_query",
    history_messages_key="chat_history"
)


# Step 7: Test the assistant with three customer queries

# These are sample test queries. Change them if your assignment
# provides different test queries.
test_cases = [
    {
        "customer_query": "Where is my order? It has not arrived yet.",
        "expected_response": "Please share your order ID so I can check the status of your order."
    },
    {
        "customer_query": "I am frustrated because this is the second time my return request has not been resolved.",
        "expected_response": "I am sorry for the inconvenience. I can help you with your return request. Please share your order ID."
    },
    {
        "customer_query": "Can you tell me more about the product?",
        "expected_response": "Please share the product name so I can provide the correct information."
    }
]

session_id = "shopease_session_001"

config = {
    "configurable": {
        "session_id": session_id
    }
}

for turn, test in enumerate(test_cases, start=1):
    result = assistant.invoke(
        {
            "customer_query": test["customer_query"],
            "example_query": example_query,
            "example_response": example_response
        },
        config=config
    )

    print(f"\nTurn {turn}")
    print("Customer query:", test["customer_query"])
    print("Assistant response:", result)
    print("Expected response:", test["expected_response"])


# Step 8: Display the stored conversation history

print("\nQ1 - Conversation History")

for message in store[session_id].messages:
    print(f"{message.type}: {message.content}")


# Step 9: Detect customer frustration from the conversation

frustration_words = [
    "still not resolved",
    "again",
    "second time",
    "frustrated",
    "escalate",
    "manager",
    "unacceptable",
    "ridiculous"
]

frustration_count = 0

for message in store[session_id].messages:
    if message.type == "human":
        customer_message = message.content.lower()

        if any(word in customer_message for word in frustration_words):
            frustration_count += 1

print("\nNumber of frustrated customer messages:", frustration_count)


# ============================================================
# Q2: RAG-BASED CUSTOMER SUPPORT ASSISTANT
# ============================================================

print("\n" + "=" * 60)
print("Q2: RAG-BASED CUSTOMER SUPPORT ASSISTANT")
print("=" * 60)


# Step 1: Load the dataset into LangChain Documents

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import (
    create_stuff_documents_chain,
    create_retrieval_chain
)


documents = []

for idx, row in df.iterrows():
    passage = (
        f"Customer Query: {row['customer_query']}\n"
        f"Expected Response: {row['expected_response']}"
    )

    documents.append(
        Document(
            page_content=passage,
            metadata={"row_index": int(idx)}
        )
    )


# Step 2: Split documents into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

chunks = text_splitter.split_documents(documents)

print("Number of document chunks:", len(chunks))


# Step 3: Create embeddings using Amazon Bedrock

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)


# Step 4: Store the embeddings in a FAISS vector database

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)


# Step 5: Create a retriever

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Step 6: Create the RAG prompt

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are ShopEase AI Support Assistant.

            Answer the customer's question using only the information
            in the provided context.

            Context:
            <context>
            {context}
            </context>

            Instructions:
            1. Be polite and helpful.
            2. Use the retrieved context to answer.
            3. Do not make up information.
            4. If the answer is not available in the context, reply exactly:
               I don't have that information right now.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)


# Step 7: Create the document chain

docs_chain = create_stuff_documents_chain(
    llm,
    rag_prompt
)


# Step 8: Create the retrieval chain

retrieval_chain = create_retrieval_chain(
    retriever,
    docs_chain
)


# Step 9: Create session-based memory for the RAG chain

rag_store = {}


def get_rag_session_history(session_id: str):
    """Return the RAG conversation history for a session."""
    if session_id not in rag_store:
        rag_store[session_id] = InMemoryChatMessageHistory()

    return rag_store[session_id]


# Step 10: Add chat history to the RAG chain

final_rag_chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_rag_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


# Step 11: Test the RAG assistant

rag_session_id = "shopease_rag_001"

rag_config = {
    "configurable": {
        "session_id": rag_session_id
    }
}

# The following examples use dataframe positions 12 and 14.
# iloc[12] means the 13th data row; iloc[14] means the 15th data row.
# Change these positions if your assignment uses different rows.

query_1 = str(df.iloc[12]["customer_query"])
query_2 = str(df.iloc[14]["customer_query"])

rag_test_queries = [
    query_1,
    query_2,
    "Can you explain that in more detail?"
]

for turn, question in enumerate(rag_test_queries, start=1):
    result = final_rag_chain.invoke(
        {"input": question},
        config=rag_config
    )

    print(f"\nRAG Turn {turn}")
    print("Customer query:", question)
    print("Assistant response:", result["answer"])

    # Show the retrieved source passages
    if "context" in result:
        print("\nRetrieved context:")
        for doc in result["context"]:
            print(doc.page_content)
            print("-" * 40)


# Step 12: Display the RAG conversation history

print("\nQ2 - RAG Conversation History")

for message in rag_store[rag_session_id].messages:
    print(f"{message.type}: {message.content}")


# ============================================================
# END OF ASSIGNMENT
# ============================================================


# IMPORTANT NOTE:
# RunnableWithMessageHistory stores previous messages. However,
# create_retrieval_chain does not automatically convert a follow-up
# question into a standalone search query.
#
# For example:
#   Turn 1: "Tell me about Galaxy S25 FE."
#   Turn 2: "What about its battery?"
#
# A history-aware retriever is a better solution for this type of
# follow-up question. The code above follows the basic RAG chain
# structure used in the assignment.
#
# Also check that the selected Bedrock model IDs are enabled in
# your AWS account and available in your chosen AWS region.
