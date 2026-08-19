import re


def is_definition(sentence: str) -> bool:
    '''Determines if a sentence is an definition of a concept or idea'''
    sentence = sentence.lower()

    #Multi-word phrases: use basic substring check
    def_cues = ['definition', 'known as', 'is called', 'means that', 'refers to', 'in other words']
    for cue in def_cues:
        if cue in sentence:
            return True
    
    #Regrex to avoid mismatches within a word
    return bool(re.search(r'\b(define|denoted|called)\b', sentence))



def is_example(sentence: str) -> bool:
    '''Determines if a sentence is an example of a concept or idea'''
    sentence = sentence.lower()

    example_cues = ['example', 'for instance', "let's say", "let's work through"]
    for cue in example_cues:
        if cue in sentence:
            return True
    
    return bool(re.search(r'\b(suppose|imagine|consider)\b', sentence))


def is_question(sentence: str) -> bool:
    '''Determines if a sentence is an question or practice problem of a concept or idea'''
    sentence = sentence.lower()

    if sentence.strip().endswith('?') or sentence.strip().startswith('why'):
        return True

    question_cues = ['what is', 'how many', 'how do', "let's solve", 'who can tell me']
    for cue in question_cues:
        if cue in sentence:
            return True
    
    return bool(re.search(r'\b(derive|calculate|compute|find)\b', sentence))

    

def classify_unit_type(sentence: str) -> str:
    '''Classifies a sentence as 'question', 'definition', 'example', or 'explanation' as a fallback'''
    if is_question(sentence):
        return 'question'
    elif is_definition(sentence):
        return 'definition'
    elif is_example(sentence):
        return 'example'
    else:
        return 'explanation'


if __name__ == '__main__':
    test_sentences = [
        "In many problems we start with an initial idea, called a prior, and then gain new evidence.",
        "Suppose a disease is rare, affecting only one percent of the population.",
        "Conditional independence asks a different question: after we already know some additional information, do the two events still provide extra information about each other?",
        "This is extremely useful in applications ranging from medical diagnosis to spam filtering and machine learning.",
    ]
    for sent in test_sentences:
        print(f"[{classify_unit_type(sent)}] {sent}")
