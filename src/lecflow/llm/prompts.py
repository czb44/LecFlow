NOTE_REFINEMENT_PROMPT = """
    Refine provided list of lecture sentences into concise notes:
    1. REMOVE insignificant, irrelevant, incoherent, repetitive, or off-topic content.
    2. KEEP sentences containing important concepts, information, definitions, examples, 
    formulas, algorithms, or results.
    3. REMOVE content that only contains speaker labels, acknowledgments, conversational exchanges, 
    audience/professor markers, or filler dialogue.
    4. PRESERVE the original meaning and order of retained content.
    5. DO NOT add new information or commentary.
    6. If no meaningful academic content remains, return an empty string.
    7. Otherwise, return the retained content as concise Markdown bullets.
"""

HOUSEKEEPING_REFINEMENT_PROMPT = """
    Filter the provided list of lecture sentences classified as housekeeping:
    1. KEEP only true lecture logistics or administrative information, such as due dates, 
    scheduling, course policies, relevant announcements, or assignment/exam information. 
    2. REMOVE academic explanations, examples, calculations, formulas, or technical discussion.
    3. REMOVE content that only contains speaker labels, acknowledgments, conversational exchanges, 
    audience/professor markers, or filler dialogue.
    4. PRESERVE the original meaning and order of retained content.
    5. DO NOT add new information or commentary.
    6. If no meaningful housekeeping content remains, return an empty string.
    7. Otherwise, return the retained sentences as concise Markdown bullets.
"""
