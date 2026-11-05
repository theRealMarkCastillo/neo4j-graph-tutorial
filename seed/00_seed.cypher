// Starting graph for Lesson 1: 5 people, 4 skills, 3 projects.
// Load into an EMPTY database:
//   docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" -f /seed/00_seed.cypher
CREATE
 (alice:Person {name:'Alice', role:'Engineer'}),
 (bob:Person {name:'Bob', role:'Designer'}),
 (cara:Person {name:'Cara', role:'Engineer'}),
 (dev:Person {name:'Dev', role:'Manager'}),
 (eve:Person {name:'Eve', role:'Data Scientist'}),
 (py:Skill {name:'Python'}), (ts:Skill {name:'TypeScript'}), (ml:Skill {name:'ML'}), (ux:Skill {name:'UX'}),
 (search:Project {name:'Search Revamp', status:'active'}),
 (app:Project {name:'Mobile App', status:'active'}),
 (legacy:Project {name:'Legacy Migration', status:'done'}),
 (alice)-[:KNOWS {since:2019}]->(bob), (alice)-[:KNOWS {since:2021}]->(cara),
 (cara)-[:KNOWS {since:2020}]->(eve), (dev)-[:MANAGES]->(alice), (dev)-[:MANAGES]->(cara),
 (alice)-[:HAS_SKILL {level:5}]->(py), (alice)-[:HAS_SKILL {level:3}]->(ts),
 (cara)-[:HAS_SKILL {level:4}]->(ts), (eve)-[:HAS_SKILL {level:5}]->(ml), (eve)-[:HAS_SKILL {level:4}]->(py),
 (bob)-[:HAS_SKILL {level:5}]->(ux),
 (alice)-[:WORKS_ON]->(search), (eve)-[:WORKS_ON]->(search), (cara)-[:WORKS_ON]->(app),
 (bob)-[:WORKS_ON]->(app), (alice)-[:WORKS_ON]->(legacy),
 (search)-[:REQUIRES]->(py), (search)-[:REQUIRES]->(ml), (app)-[:REQUIRES]->(ts), (app)-[:REQUIRES]->(ux);
