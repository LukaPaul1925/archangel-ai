from neo4j import GraphDatabase
import os
from typing import Dict, Any, List
import logging
from transformers import BertTokenizer, BertModel
from src.data.data_pipeline import TextDataLoader  # Fixed import

class CosmicKnowledgeEngine:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        self.tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.model = BertModel.from_pretrained("dmis-lab/biobert-v1.1")
        logging.basicConfig(level=logging.INFO, filename="logs/archangel.log")
        self.data_loader = TextDataLoader()  # Initialize data loader

    def extract_triplets(self, text: str) -> List[Dict[str, str]]:
        """Extract symptom-diagnosis-treatment triplets using BioBERT."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        # Simplified placeholder for triplet extraction (to be enhanced with NLP logic)
        # In practice, use a relation extraction model to identify triplets
        triplets = [
            {"symptom": "chest pain", "diagnosis": "myocardial infarction", "treatment": "aspirin"}
            if "pain" in text.lower() else {"symptom": "fever", "diagnosis": "infection", "treatment": "antibiotics"}
        ]
        logging.info(f"Extracted triplets: {triplets}")
        return triplets

    def rare_disease_score(self, diagnosis: str) -> float:
        """Calculate a score for rare disease likelihood (placeholder logic)."""
        rare_diseases = {"myocardial infarction": 0.1, "infection": 0.05}  # Example mapping
        score = rare_diseases.get(diagnosis.lower(), 0.01)  # Default 0.01 for unknown
        logging.info(f"Rare disease score for {diagnosis}: {score}")
        return score

    def query(self, symptom: str) -> Dict[str, Any]:
        """Query the knowledge graph for diagnosis and treatment."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Symptom {name: $symptom})-[r:CAUSES]->(d:Diagnosis) "
                "RETURN d.name AS diagnosis, r.treatment AS treatment",
                symptom=symptom
            )
            records = [record for record in result]
            if records:
                return {"diagnosis": records[0]["diagnosis"], "treatment": records[0]["treatment"]}
            return {"diagnosis": "Unknown", "treatment": "Consult physician"}

    def populate_graph(self, triplets: List[Dict[str, str]]):
        """Populate the knowledge graph with extracted triplets."""
        with self.driver.session() as session:
            for triplet in triplets:
                query = (
                    "MERGE (s:Symptom {name: $symptom}) "
                    "MERGE (d:Diagnosis {name: $diagnosis}) "
                    "MERGE (s)-[:CAUSES {treatment: $treatment}]->(d)"
                )
                session.run(query, symptom=triplet["symptom"], diagnosis=triplet["diagnosis"], treatment=triplet["treatment"])
            logging.info(f"Populated graph with {len(triplets)} triplets")

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()

    def test_connection(self):
        """Temporary method to test Neo4j connection."""
        with self.driver.session() as session:
            result = session.run("CREATE (n:Test {name: 'ARCHANGEL'}) RETURN n")
            print([record["n"]["name"] for record in result], flush=True)

if __name__ == "__main__":
    engine = CosmicKnowledgeEngine()
    engine.test_connection()
    engine.close()
