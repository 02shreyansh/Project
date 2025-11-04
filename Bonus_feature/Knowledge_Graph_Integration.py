import os
from typing import List, Dict, Set, Tuple, Any, Optional
import json
from datetime import datetime
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

class DocumentKnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_types = defaultdict(int)
        self.relationship_types = defaultdict(int)
        self.entity_index = {}
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any]
    ):
        self.graph.add_node(
            entity_id,
            entity_type=entity_type,
            **properties
        )
        self.entity_index[entity_id] = {
            'entity_type': entity_type,
            'properties': properties
        }
        self.entity_types[entity_type] += 1
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        if properties is None:
            properties = {}
        self.graph.add_edge(
            source_id,
            target_id,
            relationship_type=relationship_type,
            **properties
        )
        self.relationship_types[relationship_type] += 1
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        if entity_id in self.graph:
            return dict(self.graph.nodes[entity_id])
        return None
    
    def get_relationships(
        self,
        entity_id: str,
        direction: str = 'both'
    ) -> List[Dict]:
        relationships = []
        
        if direction in ['outgoing', 'both']:
            for target in self.graph.successors(entity_id):
                for key, edge_data in self.graph[entity_id][target].items():
                    relationships.append({
                        'source': entity_id,
                        'target': target,
                        'type': edge_data.get('relationship_type', 'UNKNOWN'),
                        'properties': {k: v for k, v in edge_data.items() 
                                     if k != 'relationship_type'},
                        'direction': 'outgoing'
                    })
        
        if direction in ['incoming', 'both']:
            for source in self.graph.predecessors(entity_id):
                for key, edge_data in self.graph[source][entity_id].items():
                    relationships.append({
                        'source': source,
                        'target': entity_id,
                        'type': edge_data.get('relationship_type', 'UNKNOWN'),
                        'properties': {k: v for k, v in edge_data.items() 
                                     if k != 'relationship_type'},
                        'direction': 'incoming'
                    })
        
        return relationships
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> List[List[str]]:
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_connected_entities(
        self,
        entity_id: str,
        depth: int = 2
    ) -> Set[str]:
        if entity_id not in self.graph:
            return set()
        
        connected = set([entity_id])
        current_level = {entity_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            
            connected.update(next_level)
            current_level = next_level
        
        return connected
    
    def query_by_pattern(
        self,
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        properties: Optional[Dict] = None
    ) -> List[Dict]:
        results = []
        if entity_type or properties:
            for node_id, node_data in self.graph.nodes(data=True):
                match = True
                
                if entity_type and node_data.get('entity_type') != entity_type:
                    match = False
                
                if properties:
                    for key, value in properties.items():
                        if node_data.get(key) != value:
                            match = False
                            break
                
                if match:
                    results.append({
                        'entity_id': node_id,
                        'entity_type': node_data.get('entity_type'),
                        'properties': {k: v for k, v in node_data.items() 
                                     if k != 'entity_type'}
                    })
        if relationship_type:
            edge_results = []
            for source, target, edge_data in self.graph.edges(data=True):
                if edge_data.get('relationship_type') == relationship_type:
                    edge_results.append({
                        'source': source,
                        'target': target,
                        'relationship_type': relationship_type,
                        'properties': {k: v for k, v in edge_data.items() 
                                     if k != 'relationship_type'}
                    })
            
            if not entity_type and not properties:
                return edge_results
        
        return results
    
    def get_statistics(self) -> Dict:
        return {
            'total_entities': self.graph.number_of_nodes(),
            'total_relationships': self.graph.number_of_edges(),
            'entity_types': dict(self.entity_types),
            'relationship_types': dict(self.relationship_types),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
        }
    
    def export_to_dict(self) -> Dict:
        return {
            'nodes': [
                {
                    'id': node_id,
                    **node_data
                }
                for node_id, node_data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': source,
                    'target': target,
                    **edge_data
                }
                for source, target, edge_data in self.graph.edges(data=True)
            ]
        }
    
    def save_to_file(self, filepath: str):
        data = self.export_to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Knowledge graph saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.graph.clear()
        self.entity_index.clear()
        self.entity_types.clear()
        self.relationship_types.clear()
        for node in data['nodes']:
            node_id = node.pop('id')
            entity_type = node.pop('entity_type', 'Unknown')
            self.add_entity(node_id, entity_type, node)
        for edge in data['edges']:
            source = edge.pop('source')
            target = edge.pop('target')
            rel_type = edge.pop('relationship_type', 'RELATED_TO')
            self.add_relationship(source, target, rel_type, edge)
        
        print(f"✓ Knowledge graph loaded from {filepath}")


class DocumentKnowledgeGraphBuilder:
    def __init__(self):
        self.kg = DocumentKnowledgeGraph()
        self.document_entities = {}
    
    def process_document(
        self,
        doc_id: str,
        document_type: str,
        extracted_fields: Dict,
        entities: Dict
    ):
        print(f"\nProcessing document: {doc_id} ({document_type})")
        self.kg.add_entity(
            doc_id,
            f'Document_{document_type}',
            {
                'document_type': document_type,
                'timestamp': datetime.now().isoformat(),
                **extracted_fields
            }
        )
        self.document_entities[doc_id] = {
            'type': document_type,
            'fields': extracted_fields,
            'entities': entities
        }
        if document_type == 'invoice':
            self._process_invoice(doc_id, extracted_fields, entities)
        elif document_type == 'resume':
            self._process_resume(doc_id, extracted_fields, entities)
        elif document_type == 'report':
            self._process_report(doc_id, extracted_fields, entities)
        
        print(f"✓ Document {doc_id} processed")
    
    def _process_invoice(
        self,
        doc_id: str,
        fields: Dict,
        entities: Dict
    ):
        if 'vendor' in fields:
            vendor_id = f"ORG_{fields['vendor'].replace(' ', '_')}"
            self.kg.add_entity(
                vendor_id,
                'Organization',
                {
                    'name': fields['vendor'],
                    'type': 'vendor'
                }
            )
            self.kg.add_relationship(
                doc_id,
                vendor_id,
                'ISSUED_BY',
                {'timestamp': datetime.now().isoformat()}
            )
        for org in entities.get('organizations', []):
            org_id = f"ORG_{org.replace(' ', '_')}"
            if not self.kg.get_entity(org_id):
                self.kg.add_entity(
                    org_id,
                    'Organization',
                    {'name': org}
                )
            self.kg.add_relationship(
                doc_id,
                org_id,
                'MENTIONS',
                {}
            )
        if 'date' in fields:
            date_id = f"DATE_{fields['date']}"
            self.kg.add_entity(
                date_id,
                'Date',
                {'date': fields['date']}
            )
            self.kg.add_relationship(
                doc_id,
                date_id,
                'DATED',
                {}
            )
        if 'total_amount' in fields:
            amount_id = f"AMOUNT_{doc_id}"
            self.kg.add_entity(
                amount_id,
                'MonetaryValue',
                {'amount': fields['total_amount']}
            )
            self.kg.add_relationship(
                doc_id,
                amount_id,
                'HAS_AMOUNT',
                {}
            )
    
    def _process_resume(
        self,
        doc_id: str,
        fields: Dict,
        entities: Dict
    ):
        person_name = entities.get('persons', [None])[0]
        if person_name:
            person_id = f"PERSON_{person_name.replace(' ', '_')}"
            self.kg.add_entity(
                person_id,
                'Person',
                {
                    'name': person_name,
                    'email': fields.get('email'),
                    'phone': fields.get('phone')
                }
            )
            self.kg.add_relationship(
                doc_id,
                person_id,
                'DESCRIBES',
                {}
            )
            skills = fields.get('skills', [])
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(',')]
            
            for skill in skills:
                skill_id = f"SKILL_{skill.replace(' ', '_')}"
                if not self.kg.get_entity(skill_id):
                    self.kg.add_entity(
                        skill_id,
                        'Skill',
                        {'name': skill}
                    )
                self.kg.add_relationship(
                    person_id,
                    skill_id,
                    'HAS_SKILL',
                    {}
                )
            for org in entities.get('organizations', []):
                org_id = f"ORG_{org.replace(' ', '_')}"
                if not self.kg.get_entity(org_id):
                    self.kg.add_entity(
                        org_id,
                        'Organization',
                        {'name': org}
                    )
                self.kg.add_relationship(
                    person_id,
                    org_id,
                    'WORKED_AT',
                    {}
                )
    
    def _process_report(
        self,
        doc_id: str,
        fields: Dict,
        entities: Dict
    ):
        for org in entities.get('organizations', []):
            org_id = f"ORG_{org.replace(' ', '_')}"
            if not self.kg.get_entity(org_id):
                self.kg.add_entity(
                    org_id,
                    'Organization',
                    {'name': org}
                )
            self.kg.add_relationship(
                doc_id,
                org_id,
                'MENTIONS',
                {}
            )
        for person in entities.get('persons', []):
            person_id = f"PERSON_{person.replace(' ', '_')}"
            if not self.kg.get_entity(person_id):
                self.kg.add_entity(
                    person_id,
                    'Person',
                    {'name': person}
                )
            self.kg.add_relationship(
                doc_id,
                person_id,
                'MENTIONS',
                {}
            )
    
    def find_connections(
        self,
        entity_id: str,
        depth: int = 2
    ) -> Dict:
        """Find all connections for an entity"""
        
        connected = self.kg.get_connected_entities(entity_id, depth)
        subgraph = self.kg.graph.subgraph(connected)
        
        return {
            'entity_id': entity_id,
            'connected_entities': list(connected),
            'total_connections': len(connected) - 1,
            'relationships': [
                {
                    'source': source,
                    'target': target,
                    'type': data.get('relationship_type', 'UNKNOWN')
                }
                for source, target, data in subgraph.edges(data=True)
            ]
        }
    
    def get_entity_insights(self, entity_id: str) -> Dict:
        entity = self.kg.get_entity(entity_id)
        if not entity:
            return {'error': 'Entity not found'}
        
        relationships = self.kg.get_relationships(entity_id)
        rel_type_counts = defaultdict(int)
        for rel in relationships:
            rel_type_counts[rel['type']] += 1
        connected_types = defaultdict(int)
        for rel in relationships:
            target_id = rel['target'] if rel['direction'] == 'outgoing' else rel['source']
            target_entity = self.kg.get_entity(target_id)
            if target_entity:
                connected_types[target_entity.get('entity_type', 'Unknown')] += 1
        
        return {
            'entity_id': entity_id,
            'entity_type': entity.get('entity_type'),
            'properties': {k: v for k, v in entity.items() if k != 'entity_type'},
            'total_relationships': len(relationships),
            'relationship_types': dict(rel_type_counts),
            'connected_entity_types': dict(connected_types)
        }

class KnowledgeGraphVisualizer:
    def __init__(self, kg: DocumentKnowledgeGraph):
        self.kg = kg
    
    def visualize_matplotlib(
        self,
        save_path: str = 'knowledge_graph.png',
        figsize: Tuple[int, int] = (16, 12)
    ):
        if self.kg.graph.number_of_nodes() == 0:
            print("⚠ Graph is empty")
            return
        pos = nx.spring_layout(self.kg.graph, k=2, iterations=50)
        fig, ax = plt.subplots(figsize=figsize)
        entity_types = set(nx.get_node_attributes(self.kg.graph, 'entity_type').values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(entity_types)))
        color_map = dict(zip(entity_types, colors))
        for entity_type in entity_types:
            nodes = [n for n, d in self.kg.graph.nodes(data=True) 
                    if d.get('entity_type') == entity_type]
            
            nx.draw_networkx_nodes(
                self.kg.graph,
                pos,
                nodelist=nodes,
                node_color=[color_map[entity_type]],
                node_size=1000,
                alpha=0.8,
                label=entity_type,
                ax=ax
            )
        nx.draw_networkx_edges(
            self.kg.graph,
            pos,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=20,
            ax=ax
        )
        labels = {}
        for node, data in self.kg.graph.nodes(data=True):
            label = data.get('name', node)
            if len(label) > 20:
                label = label[:17] + '...'
            labels[node] = label
        
        nx.draw_networkx_labels(
            self.kg.graph,
            pos,
            labels,
            font_size=8,
            font_weight='bold',
            ax=ax
        )
        edge_labels = {}
        for source, target, data in self.kg.graph.edges(data=True):
            rel_type = data.get('relationship_type', '')
            edge_labels[(source, target)] = rel_type
        
        nx.draw_networkx_edge_labels(
            self.kg.graph,
            pos,
            edge_labels,
            font_size=6,
            ax=ax
        )
        
        plt.title('Document Knowledge Graph', fontsize=16, fontweight='bold')
        plt.legend(scatterpoints=1, loc='upper right', fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Graph visualization saved to {save_path}")
        plt.close()
    
    def visualize_interactive(
        self,
        save_path: str = 'knowledge_graph_interactive.html'
    ):
        net = Network(
            height='800px',
            width='100%',
            directed=True,
            notebook=True
        )
        entity_types = set(nx.get_node_attributes(self.kg.graph, 'entity_type').values())
        colors = ['#%02x%02x%02x' % tuple(int(c*255) for c in plt.cm.Set3(i)[:3]) 
                 for i in np.linspace(0, 1, len(entity_types))]
        color_map = dict(zip(entity_types, colors))
        for node_id, node_data in self.kg.graph.nodes(data=True):
            entity_type = node_data.get('entity_type', 'Unknown')
            label = node_data.get('name', node_id)
            
            net.add_node(
                node_id,
                label=label,
                color=color_map.get(entity_type, '#gray'),
                title=f"{entity_type}: {json.dumps(node_data, indent=2)}",
                size=25
            )
        
        # Add edges
        for source, target, edge_data in self.kg.graph.edges(data=True):
            rel_type = edge_data.get('relationship_type', '')
            net.add_edge(
                source,
                target,
                label=rel_type,
                title=json.dumps(edge_data, indent=2)
            )
        
        # Configure physics
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        
        # Save
        net.save_graph(save_path)
        print(f"Interactive graph saved to {save_path}")
if __name__ == "__main__":
    print("="*80)
    print("KNOWLEDGE GRAPH INTEGRATION")
    print("="*80)
    
    # Initialize builder
    builder = DocumentKnowledgeGraphBuilder()
    
    # Example: Add sample documents
    print("\nAdding sample documents...")
    
    # Invoice
    builder.process_document(
        'INV001',
        'invoice',
        {
            'invoice_no': 'INV-2025-321',
            'vendor': 'ABC Solutions Pvt Ltd',
            'total_amount': '$58,400',
            'date': '01/15/2025'
        },
        {
            'organizations': ['ABC Solutions Pvt Ltd'],
            'money': ['$58,400'],
            'dates': ['01/15/2025']
        }
    )
    
    # Resume
    builder.process_document(
        'RES001',
        'resume',
        {
            'email': 'john.doe@email.com',
            'phone': '555-0123',
            'skills': ['Python', 'Machine Learning', 'TensorFlow']
        },
        {
            'persons': ['John Doe'],
            'organizations': ['Tech Corp', 'AI Labs']
        }
    )
    
    stats = builder.kg.get_statistics()
    visualizer = KnowledgeGraphVisualizer(builder.kg)
    visualizer.visualize_matplotlib('kg_visualization.png')
    visualizer.visualize_interactive('kg_interactive.html')