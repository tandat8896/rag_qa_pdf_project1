def extract_knowledge_base_info(question: str, knowledge_base) -> str:
    if not knowledge_base:
        return ""
    relevant_entities = []
    question_lower = question.lower()
    keywords = {
        'method': ['method', 'approach', 'framework', 'technique', 'algorithm'],
        'dataset': ['dataset', 'data', 'evaluated', 'tested', 'benchmark'],
        'result': ['result', 'performance', 'accuracy', 'score', 'achieved', 'bleu', 'f1'],
        'concept': ['concept', 'learning', 'adaptation', 'training', 'supervision'],
        'future': ['future', 'work', 'direction', 'suggest', 'propose']
    }
    for entity in knowledge_base.entities.values():
        entity_lower = entity.name.lower()
        if any(word in entity_lower for word in question_lower.split()):
            relevant_entities.append(entity)
            continue
        for entity_type, type_keywords in keywords.items():
            if entity.entity_type.value.lower() in entity_type and any(kw in question_lower for kw in type_keywords):
                relevant_entities.append(entity)
                break
    if not relevant_entities:
        return ""
    kb_info = []
    entities_by_type = {}
    for entity in relevant_entities:
        entity_type = entity.entity_type.value
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity)
    for entity_type, entities in entities_by_type.items():
        if entity_type == "Paper":
            continue
        kb_info.append(f"**{entity_type}:**")
        for entity in entities:
            kb_info.append(f"- {entity.name}")
            connected = knowledge_base.get_connected_entities(entity.id)
            if connected:
                for connected_entity, relation in connected:
                    if relation.relation_type.value in ['evaluated_on', 'achieves', 'uses']:
                        kb_info.append(f"  - {relation.relation_type.value}: {connected_entity.name}")
        kb_info.append("")
    return "\n".join(kb_info)

def cosine_sim(a, b):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) 