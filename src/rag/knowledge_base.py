import re
import json
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
import spacy
from collections import defaultdict

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
    nlp = None

class EntityType(Enum):
    PAPER = "Paper"
    METHOD = "Method"
    CONCEPT = "Concept"
    DATASET = "Dataset"
    RESULT = "Result"
    RESEARCH_GAP = "ResearchGap"
    FUTURE_WORK = "FutureWork"
    IMPLEMENTATION = "Implementation"
    AUGMENTATION = "Augmentation"

class RelationType(Enum):
    INTRODUCES = "introduces"
    USES = "uses"
    EVALUATED_ON = "evaluated_on"
    ACHIEVES = "achieves"
    CITES = "cites"
    IMPROVES = "improves"
    CONTRIBUTES_TO = "contributes_to"
    RELATES_TO = "relates_to"
    ADDRESSES = "addresses"
    SUGGESTS = "suggests"

@dataclass
class Entity:
    id: str
    name: str
    entity_type: EntityType
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: RelationType
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class KnowledgeBase:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.graph = nx.DiGraph()
        self.paper_id = None
        
    def add_entity(self, entity: Entity) -> str:
        """Add entity to knowledge base"""
        if entity.id not in self.entities:
            self.entities[entity.id] = entity
            self.graph.add_node(entity.id, **asdict(entity))
        return entity.id
    
    def add_relation(self, relation: Relation) -> None:
        """Add relation to knowledge base"""
        if relation.source_id in self.entities and relation.target_id in self.entities:
            self.relations.append(relation)
            self.graph.add_edge(
                relation.source_id, 
                relation.target_id, 
                relation_type=relation.relation_type.value,
                **relation.metadata
            )
    
    def extract_paper_info(self, text: str) -> Dict:
        """Extract basic paper information"""
        # Extract title (usually in first few lines)
        lines = text.split('\n')
        title = ""
        authors = ""
        abstract = ""
        
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            line = line.strip()
            if not title and len(line) > 10 and len(line) < 200:
                # Simple heuristic for title
                if not any(word in line.lower() for word in ['abstract', 'introduction', 'conclusion']):
                    title = line
            elif 'abstract' in line.lower():
                # Extract abstract
                abstract_start = i + 1
                for j in range(abstract_start, min(abstract_start + 10, len(lines))):
                    if 'introduction' in lines[j].lower():
                        break
                    abstract += lines[j] + " "
                break
        
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract.strip()
        }
    
    def extract_methods(self, text: str) -> List[Dict]:
        """Extract methods from text, return list of dict with name, source_sentence, start_pos, end_pos, entity_type"""
        methods = []
        # Blacklists
        funding_keywords = ["fund", "grant", "program", "project", "sponsor", "foundation", "training"]
        framework_keywords = ["pytorch", "tensorflow", "scikit-learn", "keras", "mxnet", "caffe", "paddlepaddle"]
        augmentation_keywords = ["randomcrop", "resize", "flip", "augmentation", "normalize", "random", "crop", "rotation", "colorjitter"]
        hardware_keywords = ["p100", "gpu", "cpu", "ram", "tpu", "v100", "a100", "nvidia", "intel", "amd"]
        hyperparam_keywords = ["dropout", "relu", "byte", "layer", "batch", "epoch", "learning rate", "lr", "weight decay", "optimizer", "adam", "sgd", "momentum", "ms", "mb", "precision", "float", "int", "activation", "hidden size", "embedding size"]
        # Whitelist keywords for method
        method_whitelist = ["attention", "transformer", "encoding", "network", "feed-forward", "self-attention", "multi-head", "positional", "method", "approach", "framework", "model", "architecture", "block", "layer", "module"]
        # 1. spaCy NER (if available)
        if nlp:
            doc = nlp(text)
            for sent in doc.sents:
                for ent in sent.ents:
                    ent_name = ent.text.strip()
                    ent_name_lower = ent_name.lower()
                    # Loại bỏ PERSON
                    if ent.label_ == "PERSON":
                        continue
                    # Lọc funding/framework/hardware/hyperparam
                    if any(k in ent_name_lower for k in funding_keywords + framework_keywords + hardware_keywords + hyperparam_keywords):
                        continue
                    # Lọc noise: số, ký hiệu, quá ngắn, chỉ số/ký tự đặc biệt
                    if len(ent_name) < 4 or not any(c.isalpha() for c in ent_name):
                        continue
                    if ent_name.isdigit() or all(not c.isalnum() for c in ent_name):
                        continue
                    # Chỉ giữ nếu tên chứa từ khóa whitelist
                    if not any(k in ent_name_lower for k in method_whitelist):
                        continue
                    # Augmentation: phân loại riêng
                    if any(k in ent_name_lower for k in augmentation_keywords):
                        entity_type = EntityType.AUGMENTATION
                    else:
                        entity_type = EntityType.METHOD
                    # Heuristic: method names are often labeled as ORG, WORK_OF_ART, or not labeled
                    if ent.label_ in ["ORG", "WORK_OF_ART"] or ("method" in sent.text.lower() or "approach" in sent.text.lower()):
                        if 5 < len(ent_name) < 100:
                            methods.append({
                                "name": ent_name,
                                "source_sentence": sent.text.strip(),
                                "start_pos": ent.start_char,
                                "end_pos": ent.end_char,
                                "entity_type": entity_type
                            })
        # 2. Regex fallback (with sentence context)
        method_patterns = [
            r'we propose\s+([^.]*)',
            r'we introduce\s+([^.]*)',
            r'we present\s+([^.]*)',
            r'we develop\s+([^.]*)',
            r'we design\s+([^.]*)',
            r'our method\s+([^.]*)',
            r'our approach\s+([^.]*)',
            r'our framework\s+([^.]*)',
            r'our model\s+([^.]*)',
        ]
        for pattern in method_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                method_name = match.group(1).strip()
                method_name_lower = method_name.lower()
                if 5 < len(method_name) < 100:
                    # Lọc funding/framework/hardware/hyperparam
                    if any(k in method_name_lower for k in funding_keywords + framework_keywords + hardware_keywords + hyperparam_keywords):
                        continue
                    # Lọc noise: số, ký hiệu, quá ngắn, chỉ số/ký tự đặc biệt
                    if len(method_name) < 4 or not any(c.isalpha() for c in method_name):
                        continue
                    if method_name.isdigit() or all(not c.isalnum() for c in method_name):
                        continue
                    # Chỉ giữ nếu tên chứa từ khóa whitelist
                    if not any(k in method_name_lower for k in method_whitelist):
                        continue
                    # Augmentation: phân loại riêng
                    if any(k in method_name_lower for k in augmentation_keywords):
                        entity_type = EntityType.AUGMENTATION
                    else:
                        entity_type = EntityType.METHOD
                    # Find sentence containing this match
                    start = match.start(1)
                    end = match.end(1)
                    sent_start = text.rfind('.', 0, start) + 1
                    sent_end = text.find('.', end)
                    if sent_end == -1:
                        sent_end = len(text)
                    source_sentence = text[sent_start:sent_end].strip()
                    methods.append({
                        "name": method_name,
                        "source_sentence": source_sentence,
                        "start_pos": start,
                        "end_pos": end,
                        "entity_type": entity_type
                    })
        # Remove duplicates by name
        seen = set()
        unique_methods = []
        for m in methods:
            if m["name"].lower() not in seen:
                unique_methods.append(m)
                seen.add(m["name"].lower())
        return unique_methods
    
    def extract_datasets(self, text: str) -> List[Dict]:
        """Extract datasets from text, return list of dict with name, source_sentence, start_pos, end_pos, entity_type"""
        datasets = []
        # Blacklists
        funding_keywords = ["fund", "grant", "program", "project", "sponsor", "foundation", "training"]
        framework_keywords = ["pytorch", "tensorflow", "scikit-learn", "keras", "mxnet", "caffe", "paddlepaddle"]
        augmentation_keywords = ["randomcrop", "resize", "flip", "augmentation", "normalize", "random", "crop", "rotation", "colorjitter"]
        hardware_keywords = ["p100", "gpu", "cpu", "ram", "tpu", "v100", "a100", "nvidia", "intel", "amd"]
        hyperparam_keywords = ["dropout", "relu", "byte", "layer", "batch", "epoch", "learning rate", "lr", "weight decay", "optimizer", "adam", "sgd", "momentum", "ms", "mb", "precision", "float", "int", "activation", "hidden size", "embedding size"]
        # Whitelist keywords for dataset
        dataset_whitelist = ["dataset", "data", "corpus", "wmt", "imagenet", "cifar", "mnist", "office-home", "domainnet", "benchmark", "test set", "train set", "validation set"]
        # 1. spaCy NER (if available)
        if nlp:
            doc = nlp(text)
            for sent in doc.sents:
                for ent in sent.ents:
                    ent_name = ent.text.strip()
                    ent_name_lower = ent_name.lower()
                    # Loại bỏ PERSON
                    if ent.label_ == "PERSON":
                        continue
                    # Lọc funding/framework/hardware/hyperparam
                    if any(k in ent_name_lower for k in funding_keywords + framework_keywords + hardware_keywords + hyperparam_keywords):
                        continue
                    # Lọc noise: số, ký hiệu, quá ngắn, chỉ số/ký tự đặc biệt
                    if len(ent_name) < 4 or not any(c.isalpha() for c in ent_name):
                        continue
                    if ent_name.isdigit() or all(not c.isalnum() for c in ent_name):
                        continue
                    # Chỉ giữ nếu tên chứa từ khóa whitelist
                    if not any(k in ent_name_lower for k in dataset_whitelist):
                        continue
                    # Augmentation: phân loại riêng
                    if any(k in ent_name_lower for k in augmentation_keywords):
                        entity_type = EntityType.AUGMENTATION
                    else:
                        entity_type = EntityType.DATASET
                    # Heuristic: dataset names are often labeled as ORG, PRODUCT, or contain 'dataset/data/corpus'
                    if ent.label_ in ["ORG", "PRODUCT"] or any(x in ent_name_lower for x in ["dataset", "data", "corpus"]):
                        if 3 < len(ent_name) < 50:
                            datasets.append({
                                "name": ent_name,
                                "source_sentence": sent.text.strip(),
                                "start_pos": ent.start_char,
                                "end_pos": ent.end_char,
                                "entity_type": entity_type
                            })
        # 2. Regex fallback (with sentence context)
        dataset_patterns = [
            r'on\s+([A-Z][A-Za-z0-9\-\s]+(?:dataset|data|corpus))',
            r'using\s+([A-Z][A-Za-z0-9\-\s]+(?:dataset|data|corpus))',
            r'evaluated\s+on\s+([A-Z][A-Za-z0-9\-\s]+(?:dataset|data|corpus))',
            r'tested\s+on\s+([A-Z][A-Za-z0-9\-\s]+(?:dataset|data|corpus))',
        ]
        for pattern in dataset_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                dataset_name = match.group(1).strip()
                dataset_name_lower = dataset_name.lower()
                if 3 < len(dataset_name) < 50:
                    # Lọc funding/framework/hardware/hyperparam
                    if any(k in dataset_name_lower for k in funding_keywords + framework_keywords + hardware_keywords + hyperparam_keywords):
                        continue
                    # Lọc noise: số, ký hiệu, quá ngắn, chỉ số/ký tự đặc biệt
                    if len(dataset_name) < 4 or not any(c.isalpha() for c in dataset_name):
                        continue
                    if dataset_name.isdigit() or all(not c.isalnum() for c in dataset_name):
                        continue
                    # Chỉ giữ nếu tên chứa từ khóa whitelist
                    if not any(k in dataset_name_lower for k in dataset_whitelist):
                        continue
                    # Augmentation: phân loại riêng
                    if any(k in dataset_name_lower for k in augmentation_keywords):
                        entity_type = EntityType.AUGMENTATION
                    else:
                        entity_type = EntityType.DATASET
                    start = match.start(1)
                    end = match.end(1)
                    sent_start = text.rfind('.', 0, start) + 1
                    sent_end = text.find('.', end)
                    if sent_end == -1:
                        sent_end = len(text)
                    source_sentence = text[sent_start:sent_end].strip()
                    datasets.append({
                        "name": dataset_name,
                        "source_sentence": source_sentence,
                        "start_pos": start,
                        "end_pos": end,
                        "entity_type": entity_type
                    })
        # Remove duplicates by name
        seen = set()
        unique_datasets = []
        for d in datasets:
            if d["name"].lower() not in seen:
                unique_datasets.append(d)
                seen.add(d["name"].lower())
        return unique_datasets
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text"""
        concepts = []
        
        # Common concept patterns
        concept_patterns = [
            r'(?:using|with|via)\s+([A-Z][A-Za-z\s]+(?:learning|adaptation|transfer|supervision))',
            r'(?:based on|leveraging)\s+([A-Z][A-Za-z\s]+(?:learning|adaptation|transfer|supervision))',
            r'(?:employing|utilizing)\s+([A-Z][A-Za-z\s]+(?:learning|adaptation|transfer|supervision))',
        ]
        
        for pattern in concept_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept_name = match.group(1).strip()
                if len(concept_name) > 5 and len(concept_name) < 50:
                    concepts.append(concept_name)
        
        return list(set(concepts))
    
    def extract_results(self, text: str) -> List[str]:
        """Extract results/metrics from text"""
        results = []
        
        # Common result patterns
        result_patterns = [
            r'achieves?\s+([^.]*(?:accuracy|precision|recall|f1|score|performance)[^.]*)',
            r'reaches?\s+([^.]*(?:accuracy|precision|recall|f1|score|performance)[^.]*)',
            r'obtains?\s+([^.]*(?:accuracy|precision|recall|f1|score|performance)[^.]*)',
            r'achieves?\s+([0-9.]+%?\s+(?:accuracy|precision|recall|f1|score))',
            r'reaches?\s+([0-9.]+%?\s+(?:accuracy|precision|recall|f1|score))',
        ]
        
        for pattern in result_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                result = match.group(1).strip()
                if len(result) > 5 and len(result) < 100:
                    results.append(result)
        
        return list(set(results))
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citations from text"""
        citations = []
        
        # Citation patterns
        citation_patterns = [
            r'\[([0-9]+)\]',  # [1], [2], etc.
            r'\(([A-Za-z]+\s+et\s+al\.?\s*[0-9]+)\)',  # (Author et al. 2023)
            r'([A-Za-z]+\s+et\s+al\.?\s*\([0-9]+\))',  # Author et al. (2023)
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = match.group(1).strip()
                citations.append(citation)
        
        return list(set(citations))
    
    def extract_research_gaps(self, text: str) -> List[str]:
        """Extract research gaps from text"""
        gaps = []
        
        gap_patterns = [
            r'however\s+([^.]*(?:limitation|gap|challenge|issue)[^.]*)',
            r'nevertheless\s+([^.]*(?:limitation|gap|challenge|issue)[^.]*)',
            r'despite\s+([^.]*(?:limitation|gap|challenge|issue)[^.]*)',
            r'one\s+(?:major|key|main)\s+(?:limitation|gap|challenge|issue)\s+is\s+([^.]*)',
        ]
        
        for pattern in gap_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                gap = match.group(1).strip()
                if len(gap) > 10 and len(gap) < 200:
                    gaps.append(gap)
        
        return list(set(gaps))
    
    def extract_future_work(self, text: str) -> List[str]:
        """Extract future work suggestions"""
        future_work = []
        
        future_patterns = [
            r'future\s+work\s+([^.]*)',
            r'future\s+direction\s+([^.]*)',
            r'future\s+research\s+([^.]*)',
            r'we\s+plan\s+to\s+([^.]*)',
            r'future\s+studies\s+([^.]*)',
        ]
        
        for pattern in future_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                future = match.group(1).strip()
                if len(future) > 10 and len(future) < 200:
                    future_work.append(future)
        
        return list(set(future_work))
    
    def build_knowledge_base(self, documents: List[Document]) -> None:
        """Build knowledge base from documents, use metadata from extract_*"""
        if not documents:
            return
        full_text = "\n\n".join([doc.page_content for doc in documents])
        paper_info = self.extract_paper_info(full_text)
        paper_id = f"paper_{len(self.entities) + 1}"
        paper_entity = Entity(
            id=paper_id,
            name=paper_info.get("title", "Unknown Paper"),
            entity_type=EntityType.PAPER,
            metadata={
                "title": paper_info.get("title", ""),
                "authors": paper_info.get("authors", ""),
                "abstract": paper_info.get("abstract", "")
            }
        )
        self.paper_id = paper_id
        self.add_entity(paper_entity)
        # Methods & Augmentation
        methods = self.extract_methods(full_text)
        method_names = [str(m["name"]) for m in methods if "name" in m and isinstance(m["name"], str)]
        for m in methods:
            method_id = f"method_{len(self.entities) + 1}"
            method_entity = Entity(
                id=method_id,
                name=m["name"],
                entity_type=m["entity_type"],
                metadata={
                    "source_sentence": m["source_sentence"],
                    "start_pos": m["start_pos"],
                    "end_pos": m["end_pos"]
                }
            )
            self.add_entity(method_entity)
            self.add_relation(Relation(
                source_id=paper_id,
                target_id=method_id,
                relation_type=RelationType.INTRODUCES
            ))
        # Datasets & Augmentation
        datasets = self.extract_datasets(full_text)
        dataset_names = [str(d["name"]) for d in datasets if "name" in d and isinstance(d["name"], str)]
        for d in datasets:
            dataset_id = f"dataset_{len(self.entities) + 1}"
            dataset_entity = Entity(
                id=dataset_id,
                name=d["name"],
                entity_type=d["entity_type"],
                metadata={
                    "source_sentence": d["source_sentence"],
                    "start_pos": d["start_pos"],
                    "end_pos": d["end_pos"]
                }
            )
            self.add_entity(dataset_entity)
            # Link to methods if available
            method_entities = [e for e in self.entities.values() if e.entity_type == EntityType.METHOD]
            for method_entity in method_entities:
                if method_entity.name and any(isinstance(mn, str) and method_entity.name.lower() in mn.lower() for mn in method_names):
                    self.add_relation(Relation(
                        source_id=method_entity.id,
                        target_id=dataset_id,
                        relation_type=RelationType.EVALUATED_ON
                    ))
        # Extract concepts
        concepts = self.extract_concepts(full_text)
        for i, concept in enumerate(concepts):
            concept_id = f"concept_{len(self.entities) + 1}"
            concept_entity = Entity(
                id=concept_id,
                name=concept,
                entity_type=EntityType.CONCEPT
            )
            self.add_entity(concept_entity)
            self.add_relation(Relation(
                source_id=paper_id,
                target_id=concept_id,
                relation_type=RelationType.USES
            ))
        # Extract results
        results = self.extract_results(full_text)
        for i, result in enumerate(results):
            result_id = f"result_{len(self.entities) + 1}"
            result_entity = Entity(
                id=result_id,
                name=result,
                entity_type=EntityType.RESULT
            )
            self.add_entity(result_entity)
            # Link to methods
            method_entities = [e for e in self.entities.values() if e.entity_type == EntityType.METHOD]
            for method_entity in method_entities:
                if method_entity.name and any(isinstance(mn, str) and method_entity.name.lower() in mn.lower() for mn in method_names):
                    self.add_relation(Relation(
                        source_id=method_entity.id,
                        target_id=result_id,
                        relation_type=RelationType.ACHIEVES
                    ))
        # Extract research gaps
        gaps = self.extract_research_gaps(full_text)
        for i, gap in enumerate(gaps):
            gap_id = f"gap_{len(self.entities) + 1}"
            gap_entity = Entity(
                id=gap_id,
                name=gap,
                entity_type=EntityType.RESEARCH_GAP
            )
            self.add_entity(gap_entity)
            self.add_relation(Relation(
                source_id=paper_id,
                target_id=gap_id,
                relation_type=RelationType.ADDRESSES
            ))
        # Extract future work
        future_works = self.extract_future_work(full_text)
        for i, future in enumerate(future_works):
            future_id = f"future_{len(self.entities) + 1}"
            future_entity = Entity(
                id=future_id,
                name=future,
                entity_type=EntityType.FUTURE_WORK
            )
            self.add_entity(future_entity)
            self.add_relation(Relation(
                source_id=paper_id,
                target_id=future_id,
                relation_type=RelationType.SUGGESTS
            ))
    
    def get_entity_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type"""
        return [entity for entity in self.entities.values() if entity.entity_type == entity_type]
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[Relation]:
        """Get all relations of a specific type"""
        return [rel for rel in self.relations if rel.relation_type == relation_type]
    
    def get_connected_entities(self, entity_id: str) -> List[Tuple[Entity, Relation]]:
        """Get all entities connected to a given entity"""
        connected = []
        for relation in self.relations:
            if relation.source_id == entity_id:
                target_entity = self.entities.get(relation.target_id)
                if target_entity:
                    connected.append((target_entity, relation))
            elif relation.target_id == entity_id:
                source_entity = self.entities.get(relation.source_id)
                if source_entity:
                    connected.append((source_entity, relation))
        return connected
    
    def to_dict(self) -> Dict:
        """Convert knowledge base to dictionary"""
        return {
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "relations": [asdict(r) for r in self.relations]
        }
    
    def from_dict(self, data: Dict) -> None:
        """Load knowledge base from dictionary"""
        self.entities = {}
        self.relations = []
        self.graph = nx.DiGraph()
        
        # Load entities
        for entity_id, entity_data in data.get("entities", {}).items():
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                entity_type=EntityType(entity_data["entity_type"]),
                metadata=entity_data.get("metadata", {})
            )
            self.entities[entity_id] = entity
            self.graph.add_node(entity_id, **asdict(entity))
        
        # Load relations
        for relation_data in data.get("relations", []):
            relation = Relation(
                source_id=relation_data["source_id"],
                target_id=relation_data["target_id"],
                relation_type=RelationType(relation_data["relation_type"]),
                metadata=relation_data.get("metadata", {})
            )
            self.relations.append(relation)
            self.graph.add_edge(
                relation.source_id,
                relation.target_id,
                relation_type=relation.relation_type.value,
                **relation.metadata
            )
    
    def visualize(self, figsize=(12, 8)):
        """Visualize the knowledge graph"""
        plt.figure(figsize=figsize)
        
        # Position nodes using spring layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes with different colors for different entity types
        node_colors = []
        for node in self.graph.nodes():
            entity_type = self.entities[node].entity_type
            if entity_type == EntityType.PAPER:
                node_colors.append('red')
            elif entity_type == EntityType.METHOD:
                node_colors.append('blue')
            elif entity_type == EntityType.DATASET:
                node_colors.append('green')
            elif entity_type == EntityType.CONCEPT:
                node_colors.append('orange')
            elif entity_type == EntityType.RESULT:
                node_colors.append('purple')
            else:
                node_colors.append('gray')
        
        # Draw the graph
        nx.draw(
            self.graph, pos,
            node_color=node_colors,
            node_size=1000,
            font_size=8,
            font_weight='bold',
            with_labels=True,
            arrows=True,
            edge_color='gray',
            arrowsize=20
        )
        
        # Add edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'relation_type')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=6)
        
        plt.title("Knowledge Graph")
        plt.tight_layout()
        return plt.gcf()
    
    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base"""
        stats = {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entity_types": defaultdict(int),
            "relation_types": defaultdict(int)
        }
        
        for entity in self.entities.values():
            stats["entity_types"][entity.entity_type.value] += 1
        
        for relation in self.relations:
            stats["relation_types"][relation.relation_type.value] += 1
        
        return dict(stats) 