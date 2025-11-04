def validate_invoice_total(text, extracted_entities):
    try:
        numbers = re.findall(r'\d+\.?\d*', text)
        numbers = [float(n) for n in numbers if float(n) > 0]

        if 'total' in extracted_entities:
            total_str = re.findall(r'\d+\.?\d*', str(extracted_entities['total']))
            if total_str:
                claimed_total = float(total_str[0])
                tolerance = claimed_total * 0.1 
                for num in numbers:
                    if abs(num - claimed_total) < tolerance:
                        return "Valid", 0.95
                return "Inconsistent", 0.60
        return "Cannot Validate", 0.50
    except:
        return "Cannot Validate", 0.50

def rank_resume(extracted_entities):
    score = 0.5 
    if 'skills' in extracted_entities:
        skills_text = str(extracted_entities['skills']).lower()
        valuable_skills = ['python', 'machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch']

        for skill in valuable_skills:
            if skill in skills_text:
                score += 0.1
    if 'experience' in extracted_entities:
        exp_text = str(extracted_entities['experience']).lower()
        if len(exp_text) > 100:
            score += 0.2

    score = min(score, 1.0)
    if score > 0.8:
        return "Highly Qualified", score
    elif score > 0.6:
        return "Qualified", score
    else:
        return "Under Qualified", score

def validate_paper(extracted_entities):
    score = 0.5
    if 'title' in extracted_entities and len(str(extracted_entities['title'])) > 10:
        score += 0.2
    if 'abstract' in extracted_entities and len(str(extracted_entities['abstract'])) > 50:
        score += 0.3
    if score > 0.8:
        return "Complete", score
    else:
        return "Incomplete", score

def apply_reasoning(document_type, text, entities):
    if document_type == 'invoice':
        return validate_invoice_total(text, entities)
    elif document_type == 'resume':
        return rank_resume(entities)
    elif document_type == 'paper':
        return validate_paper(entities)
    else:
        return "Unknown", 0.5
print("AI reasoning layer implemented!")

