"""Prompt templates for baseline and cognitive-load-aware conditions."""


def get_baseline_prompt(query: str, context: str) -> str:
    """
    Generate baseline prompt (higher cognitive load).
    
    Args:
        query: User query
        context: Retrieved context chunks
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are a helpful assistant providing advice on digital wellbeing and focus. 
Use the following context to answer the user's question. Provide a comprehensive answer based on the information provided.

Context:
{context}

Question: {query}

Answer:"""
    
    return prompt


def get_cl_aware_prompt(query: str, context: str) -> str:
    """
    Generate cognitive-load-aware prompt (structured, lower load).
    
    Args:
        query: User query
        context: Retrieved context chunks
        
    Returns:
        Formatted prompt string with structured output requirements
    """
    prompt = f"""You are a helpful assistant providing advice on digital wellbeing and focus. 
Use the following context to answer the user's question. Structure your answer clearly to reduce cognitive load.

Context:
{context}

Question: {query}

Please provide your answer in the following structured format:

## Key Strategies
- [List 3-5 key strategies as bullet points]

## Why This Helps Your Focus
[Briefly explain why these strategies work, focusing on attention and working memory benefits in 2-3 sentences]

## Try This Today
[Provide 1-2 simple, actionable steps the user can start implementing today]

Keep your language simple and clear. Avoid overwhelming the user with too much information at once.

Answer:"""
    
    return prompt

