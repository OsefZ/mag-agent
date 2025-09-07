# mag_agent/memory/graph_store.py

from neo4j import GraphDatabase
from mag_agent.config import GRAPH_DB_URI, GRAPH_DB_USER, GRAPH_DB_PASSWORD
from mag_agent.memory.schema import Node, NodeType, EdgeType
from mag_agent.memory.cache_layer import QueryCacheManager, MaterializedViewManager, PerformanceTracker
from mag_agent.memory.context_compression import HierarchicalContextCompressor, ContextItem, AdaptiveContextSelector
from mag_agent.utils.logging_config import logger
from mag_agent.utils.code_parser import CodeParser
import json
import time
from typing import Dict, List, Any, Optional, Set

class GraphStore:
    """Graph database abstraction layer for code structure storage and retrieval.
    
    Provides optimized access to Neo4j database with caching, performance monitoring,
    and hierarchical context compression for efficient code context extraction.
    """

    def __init__(self, cache_size: int = 10000, max_context_tokens: int = 4000):
        """Initialize graph store with database connection and optimization components.
        
        Args:
            cache_size: Maximum number of queries to cache
            max_context_tokens: Token limit for context compression
        """
        logger.info("Connecting to graph database...")
        try:
            # Establish Neo4j database connection
            self.driver = GraphDatabase.driver(GRAPH_DB_URI, auth=(GRAPH_DB_USER, GRAPH_DB_PASSWORD))
            self.driver.verify_connectivity()
            logger.info("Graph database connection successful.")
            
            # Set up code analysis tools
            self.code_parser = CodeParser()
            
            # Configure caching and compression systems
            self.cache_manager = QueryCacheManager(cache_size=cache_size)
            self.materialized_views = MaterializedViewManager()
            self.performance_tracker = PerformanceTracker()
            self.context_compressor = HierarchicalContextCompressor(max_tokens=max_context_tokens)
            self.context_selector = AdaptiveContextSelector()
            
            # Monitor query patterns for optimization
            self.popular_queries = {}
            
            logger.info("Performance optimization components initialized")
            
        except Exception as e:
            logger.error(f"Failed to connect to graph database: {e}")
            raise

    def close(self):
        """Close database connection and cleanup resources.
        """
        if self.driver:
            self.driver.close()
        logger.info("Graph database connection closed.")

    def add_node(self, node: Node) -> str:
        """Insert a node into the graph database.
        
        Args:
            node: Node object with label and properties
            
        Returns:
            Database ID of the created node
        """
        with self.driver.session() as session:
            # Use internal node ID for compatibility
            result = session.run(
                "CREATE (n:{} {{properties: $properties}}) RETURN id(n) AS id".format(node.label.value),
                properties=node.properties
            )
            node_id = result.single()["id"]
            logger.debug(f"Added node with ID: {node_id}")
            return node_id

    def add_edge(self, from_node_id: str, to_node_id: str, edge_type: EdgeType):
        """Create relationship between two nodes in the graph.
        
        Args:
            from_node_id: Source node identifier
            to_node_id: Target node identifier
            edge_type: Type of relationship to create
        """
        with self.driver.session() as session:
            # Match nodes by internal ID for relationship creation
            session.run(
                "MATCH (a), (b) WHERE id(a) = $from_node_id AND id(b) = $to_node_id "
                "CREATE (a)-[r:{}]->(b)".format(edge_type.value),
                from_node_id=from_node_id, to_node_id=to_node_id
            )
            logger.debug(f"Added edge of type {edge_type.value} from {from_node_id} to {to_node_id}")

    def add_function(self, file_path: str, function_name: str, 
                    code: str, line_number: int = 0, docstring: str = None) -> Dict[str, Any]:
        """Store function definition in graph database.
        
        Args:
            file_path: Source file path
            function_name: Function identifier
            code: Complete function source code
            line_number: Starting line number in source file
            docstring: Function documentation if available
            
        Returns:
            Dictionary with node creation details
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (f:Function {
                    name: $name,
                    file_path: $file_path,
                    code: $code,
                    line_number: $line_number,
                    docstring: $docstring,
                    created_at: timestamp()
                })
                RETURN id(f) AS id, f.name AS name
                """,
                name=function_name,
                file_path=file_path,
                code=code,
                line_number=line_number,
                docstring=docstring or ""
            )
            record = result.single()
            logger.debug(f"Added function node: {function_name} with ID: {record['id']}")
            
            # Clear dependent cached queries
            self.cache_manager.invalidate_pattern('functions_*')
            self.cache_manager.invalidate_pattern('subgraph_*')
            
            return {"id": record["id"], "name": record["name"], "created": True}
    
    def add_test_failure(self, file_path: str, function_name: str, 
                        error_message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store test failure information in graph database.
        
        Args:
            file_path: Test file location
            function_name: Failing test identifier
            error_message: Error details from test execution
            metadata: Additional failure context data
            
        Returns:
            Dictionary containing created node information
        """
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (tf:TestFailure {
                    name: $name,
                    file_path: $file_path,
                    error_message: $error_message,
                    metadata: $metadata,
                    created_at: timestamp()
                })
                RETURN id(tf) AS id, tf.name AS name
                """,
                name=function_name,
                file_path=file_path,
                error_message=error_message,
                metadata=json.dumps(metadata) if metadata else "{}"
            )
            record = result.single()
            logger.debug(f"Added test failure node: {function_name} with ID: {record['id']}")
            
            # Refresh caches for test failure queries
            self.cache_manager.invalidate_pattern('test_failures_*')
            self.cache_manager.invalidate_pattern('contradictions_*')
            
            return {"id": record["id"], "name": record["name"], "created": True}
    
    def find_relevant_functions(self, search_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for functions matching query criteria.
        
        Args:
            search_query: Text to search for in function names and code
            limit: Maximum number of functions to return
            
        Returns:
            List of matching function data
        """
        cache_key = f"relevant_functions_{search_query}_{limit}"
        cached = self.cache_manager.get('find_relevant', {'query': search_query, 'limit': limit})
        if cached:
            return cached
            
        with self.driver.session() as session:
            cypher_query = """
                MATCH (f:Function)
                WHERE f.name CONTAINS $search_query OR 
                      f.code CONTAINS $search_query OR 
                      f.docstring CONTAINS $search_query OR
                      f.file_path CONTAINS $search_query
                RETURN f.name AS name, 
                       f.file_path AS file_path,
                       f.code AS code,
                       f.docstring AS docstring,
                       f.line_number AS line_number,
                       id(f) AS id
                LIMIT $limit
                """
            result = session.run(
                cypher_query,
                search_query=search_query,
                limit=limit
            )
            
            functions = []
            for record in result:
                functions.append({
                    "id": record["id"],
                    "name": record["name"],
                    "file_path": record["file_path"],
                    "code": record["code"],
                    "docstring": record["docstring"],
                    "line_number": record["line_number"]
                })
            
            # Cache the result
            self.cache_manager.put('find_relevant', {'query': search_query, 'limit': limit}, functions)
            
            return functions
    
    def find_contradictions(self, function_name: str, proposed_code: str) -> List[Dict[str, Any]]:
        """
        Find potential contradictions for a proposed code change.
        
        Args:
            function_name: Name of the function being changed
            proposed_code: The proposed new code
            
        Returns:
            List of potential contradictions
        """
        cache_key = f"contradictions_{function_name}"
        cached = self.cache_manager.get('contradictions', {'function': function_name})
        if cached:
            return cached
            
        contradictions = []
        
        with self.driver.session() as session:
            # Find test failures related to this function
            result = session.run(
                """
                MATCH (tf:TestFailure)
                WHERE tf.name CONTAINS $function_name OR 
                      tf.error_message CONTAINS $function_name
                RETURN tf.name AS test_name,
                       tf.error_message AS error,
                       tf.file_path AS file_path
                """,
                function_name=function_name
            )
            
            for record in result:
                contradictions.append({
                    "type": "test_failure",
                    "test_name": record["test_name"],
                    "error": record["error"],
                    "file_path": record["file_path"],
                    "severity": "high"
                })
            
            # Find design constraints that might be violated
            result = session.run(
                """
                MATCH (dc:DesignConstraint)
                WHERE dc.constraints CONTAINS $function_name
                RETURN dc.name AS constraint_name,
                       dc.constraints AS constraints
                """,
                function_name=function_name
            )
            
            for record in result:
                contradictions.append({
                    "type": "design_constraint",
                    "constraint_name": record["constraint_name"],
                    "constraints": record["constraints"],
                    "severity": "medium"
                })
        
        # Cache the result
        self.cache_manager.put('contradictions', {'function': function_name}, contradictions)
        
        return contradictions

    def get_subgraph_by_plan(self, plan: dict, use_compression: bool = True) -> dict:
        """
        Retrieves relevant nodes from the graph based on the agent's plan.
        Enhanced with caching, performance tracking, and context compression.
        
        Args:
            plan (dict): The structured plan containing files_to_inspect and functions_to_analyze
            use_compression (bool): Whether to apply hierarchical context compression
            
        Returns:
            Dictionary containing relevant context from the graph
        """
        start_time = time.time()
        
        # Try cache first
        cache_params = {
            'files_to_inspect': plan.get('files_to_inspect', []),
            'functions_to_analyze': plan.get('functions_to_analyze', []),
            'approach': plan.get('approach', ''),
            'use_compression': use_compression
        }
        
        cached_result = self.cache_manager.get('subgraph_by_plan', cache_params)
        if cached_result is not None:
            self.performance_tracker.record_query('subgraph_by_plan_cached', time.time() - start_time)
            return cached_result
        with self.driver.session() as session:
            relevant_context = {
                "functions": [],
                "classes": [],
                "related_functions": [],
                "test_failures": [],
                "previous_patches": [],
                "constraints": []
            }

            # Get functions related to the files in the plan
            files_to_inspect = plan.get('files_to_inspect', [])
            functions_to_analyze = plan.get('functions_to_analyze', [])
            
            # Query for relevant functions and classes based on file paths
            if files_to_inspect:
                for file_path in files_to_inspect:
                    # Get functions from structured codebase
                    result = session.run(
                        "MATCH (f:Function) "
                        "WHERE f.properties.file_path CONTAINS $file_path "
                        "RETURN f.properties AS function_data "
                        "LIMIT 10",
                        file_path=file_path
                    )
                    for record in result:
                        relevant_context["functions"].append(record["function_data"])
                    
                    # Get classes from structured codebase
                    result = session.run(
                        "MATCH (c:Class) "
                        "WHERE c.properties.file_path CONTAINS $file_path "
                        "RETURN c.properties AS class_data "
                        "LIMIT 5",
                        file_path=file_path
                    )
                    for record in result:
                        relevant_context["classes"].append(record["class_data"])
                    
                    # Get legacy function data for backward compatibility
                    result = session.run(
                        "MATCH (f:Function) "
                        "WHERE f.properties.location CONTAINS $file_path "
                        "RETURN f.properties AS function_data "
                        "LIMIT 5",
                        file_path=file_path
                    )
                    for record in result:
                        relevant_context["functions"].append(record["function_data"])

            # Query for functions by name if specified
            if functions_to_analyze:
                for func_name in functions_to_analyze:
                    # Search in structured codebase first
                    result = session.run(
                        "MATCH (f:Function) "
                        "WHERE f.properties.name = $func_name OR "
                        "f.properties.signature CONTAINS $func_name "
                        "RETURN f.properties AS function_data "
                        "LIMIT 3",
                        func_name=func_name
                    )
                    for record in result:
                        relevant_context["functions"].append(record["function_data"])
                    
                    # Also search for functions that call this function
                    result = session.run(
                        "MATCH (f:Function)-[:CALLS]->(target:Function) "
                        "WHERE target.properties.name = $func_name "
                        "RETURN f.properties AS function_data "
                        "LIMIT 3",
                        func_name=func_name
                    )
                    for record in result:
                        relevant_context["related_functions"].append(record["function_data"])
                    
                    # Legacy search for backward compatibility
                    result = session.run(
                        "MATCH (f:Function) "
                        "WHERE f.properties.code CONTAINS $func_name "
                        "RETURN f.properties AS function_data "
                        "LIMIT 3",
                        func_name=func_name
                    )
                    for record in result:
                        relevant_context["functions"].append(record["function_data"])

            # Get recent test failures that might be relevant
            result = session.run(
                "MATCH (t:TestFailure) "
                "RETURN t.properties AS test_failure "
                "ORDER BY t.properties.timestamp DESC "
                "LIMIT 3"
            )
            for record in result:
                relevant_context["test_failures"].append(record["test_failure"])

            # Get previous patches for similar problems
            approach = plan.get('approach', '')
            if approach:
                result = session.run(
                    "MATCH (f:Function) "
                    "WHERE f.properties.type = 'patch_suggestion' AND "
                    "f.properties.approach_used CONTAINS $approach "
                    "RETURN f.properties AS patch_data "
                    "LIMIT 2",
                    approach=approach[:50]  # Limit search term length
                )
                for record in result:
                    relevant_context["previous_patches"].append(record["patch_data"])

            # Get related design constraints
            result = session.run(
                "MATCH (d:DesignConstraint) "
                "RETURN d.properties AS constraint_data "
                "ORDER BY d.properties.timestamp DESC "
                "LIMIT 3"
            )
            for record in result:
                relevant_context["constraints"].append(record["constraint_data"])

            logger.debug(f"Retrieved context: {len(relevant_context['functions'])} functions, "
                        f"{len(relevant_context['classes'])} classes, "
                        f"{len(relevant_context['related_functions'])} related functions, "
                        f"{len(relevant_context['test_failures'])} test failures, "
                        f"{len(relevant_context['previous_patches'])} previous patches")
            
            # Apply hierarchical context compression if enabled
            if use_compression and (relevant_context.get('functions') or relevant_context.get('classes')):
                compressed_context = self._apply_context_compression(relevant_context, plan)
                relevant_context['compressed_sections'] = compressed_context
                logger.info("Applied hierarchical context compression")
            
            # Cache the result
            execution_time = time.time() - start_time
            self.performance_tracker.record_query('subgraph_by_plan', execution_time)
            self.cache_manager.put('subgraph_by_plan', cache_params, relevant_context)
            
            return relevant_context

    def check_for_contradictions(self, proposed_patch_id: str, use_cache: bool = True) -> list:
        """
        Perform a robust Cypher query to identify conflicts between a proposed patch 
        and existing constraints, test failures, or previous successful fixes.
        Enhanced with caching for better performance.
        
        Args:
            proposed_patch_id (str): ID of the proposed patch node
            use_cache (bool): Whether to use cached results
            
        Returns:
            List of contradiction objects with details about conflicts
        """
        start_time = time.time()
        
        # Try cache first
        if use_cache:
            cache_params = {'proposed_patch_id': proposed_patch_id}
            cached_result = self.cache_manager.get('contradictions', cache_params)
            if cached_result is not None:
                self.performance_tracker.record_query('contradictions_cached', time.time() - start_time)
                return cached_result
        with self.driver.session() as session:
            # Enhanced query for more comprehensive contradiction detection
            query = """
            MATCH (patch) WHERE id(patch) = $proposed_patch_id
            
            // Find test failures that might contradict this patch
            OPTIONAL MATCH (patch)-[:RELATES_TO|TARGETS|MODIFIES*1..2]-(target_func:Function)
            OPTIONAL MATCH (tf:TestFailure)
            WHERE (target_func.properties.file_path IS NOT NULL AND tf.properties.file_path IS NOT NULL AND
                   target_func.properties.file_path = tf.properties.file_path) OR
                  (target_func.properties.name IS NOT NULL AND tf.properties.function_name IS NOT NULL AND
                   target_func.properties.name = tf.properties.function_name) OR
                  (patch.properties.modifies_functions IS NOT NULL AND tf.properties.function_name IS NOT NULL AND
                   patch.properties.modifies_functions CONTAINS tf.properties.function_name)
            
            // Find design constraints that might be violated
            OPTIONAL MATCH (dc:DesignConstraint)
            WHERE (patch.properties.instance_id IS NOT NULL AND dc.properties.instance_id IS NOT NULL AND
                   patch.properties.instance_id = dc.properties.instance_id) OR
                  (patch.properties.approach_used IS NOT NULL AND dc.properties.constraints IS NOT NULL AND
                   dc.properties.constraints CONTAINS patch.properties.approach_used) OR
                  (patch.properties.code IS NOT NULL AND dc.properties.constraints IS NOT NULL AND
                   ANY(constraint IN split(dc.properties.constraints, ',') WHERE patch.properties.code CONTAINS trim(constraint)))
            
            // Find patches that successfully fixed similar issues but this patch reverts
            OPTIONAL MATCH (successful_patch:Function)
            WHERE successful_patch.properties.type = 'patch_suggestion' AND 
                  id(successful_patch) <> $proposed_patch_id AND
                  successful_patch.properties.instance_id = patch.properties.instance_id AND
                  successful_patch.properties.status = 'successful' AND
                  (patch.properties.reverts_changes = true OR
                   (successful_patch.properties.modifies_functions IS NOT NULL AND 
                    patch.properties.modifies_functions IS NOT NULL AND
                    ANY(func IN split(successful_patch.properties.modifies_functions, ',') 
                        WHERE patch.properties.modifies_functions CONTAINS trim(func))))
            
            // Find conflicting approaches or contradictory patches
            OPTIONAL MATCH (other_patch:Function)
            WHERE other_patch.properties.type = 'patch_suggestion' AND 
                  id(other_patch) <> $proposed_patch_id AND
                  other_patch.properties.instance_id = patch.properties.instance_id AND
                  (other_patch.properties.approach_used <> patch.properties.approach_used OR
                   other_patch.properties.contradicts_patch_id = toString(id(patch)))
            
            // Find functions that have been superseded and this patch targets old versions
            OPTIONAL MATCH (old_func:Function)-[:SUPERSEDES]->(current_func:Function)
            WHERE old_func.properties.status = 'superseded' AND
                  (patch.properties.modifies_functions CONTAINS old_func.properties.name OR
                   patch.properties.code CONTAINS old_func.properties.signature)
            
            RETURN {
                test_failures: COLLECT(DISTINCT {
                    id: id(tf),
                    error: tf.properties.error,
                    file_path: tf.properties.file_path,
                    function_name: tf.properties.function_name,
                    timestamp: tf.properties.timestamp,
                    severity: CASE WHEN tf.properties.error CONTAINS 'Fatal' OR tf.properties.error CONTAINS 'Critical' THEN 'critical' ELSE 'high' END
                }),
                design_constraints: COLLECT(DISTINCT {
                    id: id(dc),
                    problem_statement: dc.properties.problem_statement,
                    constraints: dc.properties.constraints,
                    instance_id: dc.properties.instance_id,
                    violated_constraint: CASE 
                        WHEN patch.properties.approach_used IS NOT NULL AND dc.properties.constraints CONTAINS patch.properties.approach_used 
                        THEN patch.properties.approach_used 
                        ELSE 'unknown' 
                    END
                }),
                reverted_successful_fixes: COLLECT(DISTINCT {
                    id: id(successful_patch),
                    approach: successful_patch.properties.approach_used,
                    code: successful_patch.properties.code,
                    timestamp: successful_patch.properties.timestamp,
                    modifies_functions: successful_patch.properties.modifies_functions
                }),
                conflicting_patches: COLLECT(DISTINCT {
                    id: id(other_patch),
                    approach: other_patch.properties.approach_used,
                    code: other_patch.properties.code,
                    timestamp: other_patch.properties.timestamp,
                    conflict_reason: CASE 
                        WHEN other_patch.properties.contradicts_patch_id = toString(id(patch)) THEN 'explicit_contradiction'
                        WHEN other_patch.properties.approach_used <> patch.properties.approach_used THEN 'different_approach'
                        ELSE 'unknown'
                    END
                }),
                outdated_function_targets: COLLECT(DISTINCT {
                    id: id(old_func),
                    name: old_func.properties.name,
                    signature: old_func.properties.signature,
                    current_version_id: id(current_func),
                    current_signature: current_func.properties.signature
                })
            } AS contradictions
            """
            
            result = session.run(query, proposed_patch_id=proposed_patch_id)
            contradictions = []
            
            for record in result:
                contradiction_data = record["contradictions"]
                
                # Process test failures with enhanced severity
                for tf in contradiction_data.get("test_failures", []):
                    if tf.get("error"):
                        contradictions.append({
                            "type": "test_failure",
                            "severity": tf.get("severity", "high"),
                            "description": f"Test failure in {tf.get('file_path', 'unknown')}: {tf.get('error', 'Unknown error')[:200]}...",
                            "source_id": tf.get("id"),
                            "details": tf
                        })
                
                # Process design constraints with violation details
                for dc in contradiction_data.get("design_constraints", []):
                    if dc.get("problem_statement"):
                        violated = dc.get("violated_constraint", "unknown")
                        contradictions.append({
                            "type": "design_constraint", 
                            "severity": "medium",
                            "description": f"Violates design constraint ({violated}): {dc.get('problem_statement', '')[:100]}...",
                            "source_id": dc.get("id"),
                            "details": dc
                        })
                
                # Process reverted successful fixes - HIGH SEVERITY
                for sf in contradiction_data.get("reverted_successful_fixes", []):
                    if sf.get("approach"):
                        contradictions.append({
                            "type": "reverted_successful_fix",
                            "severity": "critical",
                            "description": f"Patch may revert a previously successful fix that used: {sf.get('approach', 'Unknown')}",
                            "source_id": sf.get("id"),
                            "details": sf
                        })
                
                # Process conflicting patches
                for cp in contradiction_data.get("conflicting_patches", []):
                    if cp.get("approach"):
                        reason = cp.get("conflict_reason", "unknown")
                        severity = "medium" if reason == "explicit_contradiction" else "low"
                        contradictions.append({
                            "type": "conflicting_patch",
                            "severity": severity,
                            "description": f"Conflicts with previous patch ({reason}): {cp.get('approach', 'Unknown')}",
                            "source_id": cp.get("id"),
                            "details": cp
                        })
                
                # Process outdated function targets
                for of in contradiction_data.get("outdated_function_targets", []):
                    if of.get("name"):
                        contradictions.append({
                            "type": "outdated_target",
                            "severity": "medium",
                            "description": f"Patch targets superseded function '{of.get('name')}' instead of current version",
                            "source_id": of.get("id"),
                            "details": of
                        })
            
            logger.debug(f"Found {len(contradictions)} contradictions for patch {proposed_patch_id}")
            
            # Cache the result
            execution_time = time.time() - start_time
            self.performance_tracker.record_query('contradictions', execution_time)
            
            if use_cache:
                cache_params = {'proposed_patch_id': proposed_patch_id}
                self.cache_manager.put('contradictions', cache_params, contradictions, ttl=1800)  # 30min TTL
            
            return contradictions

    def get_related_nodes(self, node_id: str, relationship_types: list = None, max_depth: int = 2) -> list:
        """
        Get nodes related to a given node within a specified depth.
        
        Args:
            node_id (str): The ID of the starting node
            relationship_types (list): List of relationship types to follow (optional)
            max_depth (int): Maximum traversal depth
            
        Returns:
            List of related node data
        """
        with self.driver.session() as session:
            if relationship_types:
                rel_filter = "|".join([f":{rel}" for rel in relationship_types])
                query = (
                    f"MATCH (start) WHERE id(start) = $node_id "
                    f"MATCH (start)-[r{rel_filter}*1..{max_depth}]-(related) "
                    f"RETURN DISTINCT related.properties AS node_data, "
                    f"labels(related) AS labels LIMIT 10"
                )
            else:
                query = (
                    f"MATCH (start) WHERE id(start) = $node_id "
                    f"MATCH (start)-[*1..{max_depth}]-(related) "
                    f"RETURN DISTINCT related.properties AS node_data, "
                    f"labels(related) AS labels LIMIT 10"
                )
            
            result = session.run(query, node_id=node_id)
            related_nodes = []
            for record in result:
                node_data = record["node_data"]
                node_data["labels"] = record["labels"]
                related_nodes.append(node_data)
            
            return related_nodes

    def ingest_codebase(self, repo_path: str, instance_id: str = None, use_incremental: bool = True) -> dict:
        """
        Parse and ingest an entire codebase into the graph with structured code information.
        
        Args:
            repo_path (str): Path to the repository to parse
            instance_id (str): Optional instance ID to associate with this codebase
            
        Returns:
            Dict with ingestion statistics and node IDs
        """
        logger.info(f"Starting codebase ingestion for: {repo_path}")
        
        # Check if this repo has already been ingested
        with self.driver.session() as session:
            existing_check = session.run(
                "MATCH (f:Function) WHERE f.properties.repo_path = $repo_path "
                "RETURN COUNT(f) as count",
                repo_path=repo_path
            )
            existing_count = existing_check.single()["count"]
            
            if existing_count > 0:
                logger.info(f"Repository {repo_path} already has {existing_count} functions in graph. Skipping ingestion.")
                return {"status": "skipped", "reason": "already_ingested", "existing_functions": existing_count}
        
        # Parse the codebase
        try:
            code_elements = self.code_parser.parse_directory(repo_path, max_files=50)  # Limit for performance
        except Exception as e:
            logger.error(f"Failed to parse codebase at {repo_path}: {e}")
            return {"status": "failed", "error": str(e)}
        
        if not code_elements:
            logger.warning(f"No code elements found in {repo_path}")
            return {"status": "empty", "elements_found": 0}
        
        # Statistics
        stats = {
            "functions_added": 0,
            "classes_added": 0,
            "calls_edges_added": 0,
            "inherits_edges_added": 0,
            "contains_edges_added": 0,
            "node_ids": []
        }
        
        # Maps to track node IDs for relationship creation
        element_to_node_id = {}
        class_to_node_id = {}
        
        with self.driver.session() as session:
            # First pass: Create all function and class nodes
            for element in code_elements:
                try:
                    # Create node properties
                    properties = {
                        "name": element.name,
                        "element_type": element.element_type,
                        "signature": element.signature,
                        "docstring": element.docstring or "",
                        "file_path": element.file_path,
                        "repo_path": repo_path,
                        "start_line": element.start_line,
                        "end_line": element.end_line,
                        "parent_class": element.parent_class
                    }
                    
                    if instance_id:
                        properties["instance_id"] = instance_id
                    
                    # Determine node type and set status for functions
                    if element.element_type == 'class':
                        node_type = NodeType.CLASS
                        stats["classes_added"] += 1
                    else:
                        node_type = NodeType.FUNCTION
                        properties["status"] = "active"  # All new functions are active by default
                        stats["functions_added"] += 1
                    
                    # Create node
                    node = Node(label=node_type, properties=properties)
                    node_id = self.add_node(node)
                    stats["node_ids"].append(node_id)
                    
                    # Store for relationship creation
                    element_key = f"{element.file_path}:{element.name}"
                    if element.parent_class:
                        element_key = f"{element.file_path}:{element.parent_class}.{element.name}"
                    
                    element_to_node_id[element_key] = node_id
                    
                    if element.element_type == 'class':
                        class_key = f"{element.file_path}:{element.name}"
                        class_to_node_id[class_key] = node_id
                    
                except Exception as e:
                    logger.error(f"Failed to create node for {element.name}: {e}")
                    continue
            
            # Second pass: Create relationships
            for element in code_elements:
                try:
                    element_key = f"{element.file_path}:{element.name}"
                    if element.parent_class:
                        element_key = f"{element.file_path}:{element.parent_class}.{element.name}"
                    
                    if element_key not in element_to_node_id:
                        continue
                        
                    current_node_id = element_to_node_id[element_key]
                    
                    # Create CONTAINS relationships for class methods
                    if element.parent_class:
                        parent_class_key = f"{element.file_path}:{element.parent_class}"
                        if parent_class_key in class_to_node_id:
                            parent_node_id = class_to_node_id[parent_class_key]
                            self.add_edge(parent_node_id, current_node_id, EdgeType.CONTAINS)
                            stats["contains_edges_added"] += 1
                    
                    # Create CALLS relationships
                    if element.calls:
                        for called_func in element.calls:
                            # Try to find the called function in our parsed elements
                            possible_keys = [
                                f"{element.file_path}:{called_func}",  # Same file
                                f"{element.file_path}:{element.parent_class}.{called_func}" if element.parent_class else None,  # Same class
                            ]
                            
                            for possible_key in possible_keys:
                                if possible_key and possible_key in element_to_node_id:
                                    target_node_id = element_to_node_id[possible_key]
                                    self.add_edge(current_node_id, target_node_id, EdgeType.CALLS)
                                    stats["calls_edges_added"] += 1
                                    break
                    
                    # Create INHERITS_FROM relationships for classes
                    if element.element_type == 'class' and element.inherits_from:
                        for base_class in element.inherits_from:
                            # Try to find the base class
                            base_class_key = f"{element.file_path}:{base_class}"
                            if base_class_key in class_to_node_id:
                                base_node_id = class_to_node_id[base_class_key]
                                self.add_edge(current_node_id, base_node_id, EdgeType.INHERITS_FROM)
                                stats["inherits_edges_added"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to create relationships for {element.name}: {e}")
                    continue
        
        # Track performance and invalidate related cache entries
        execution_time = time.time() - start_time
        self.performance_tracker.record_query('codebase_ingestion', execution_time)
        
        # Invalidate cached queries that might be affected by new code
        self.cache_manager.invalidate_pattern('subgraph_by_plan')
        self.cache_manager.invalidate_pattern('related_nodes')
        
        logger.info(f"Codebase ingestion complete in {execution_time:.2f}s. Stats: {stats}")
        return {"status": "completed", **stats, "ingestion_time": execution_time}
    
    def update_function_version(self, old_function_id: str, new_function_node: Node) -> str:
        """
        Updates a function to a new version by adding the new node, creating a SUPERSEDES edge,
        and marking the old function as superseded.
        
        Args:
            old_function_id (str): ID of the existing function node to supersede
            new_function_node (Node): New function node to add
            
        Returns:
            str: ID of the newly created function node
        """
        with self.driver.session() as session:
            # First, add the new function node
            # Ensure the new function has active status
            new_properties = new_function_node.properties.copy()
            new_properties["status"] = "active"
            new_node = Node(label=new_function_node.label, properties=new_properties)
            new_function_id = self.add_node(new_node)
            
            # Update the old function's status to superseded
            session.run(
                "MATCH (old_func) WHERE id(old_func) = $old_id "
                "SET old_func.properties.status = 'superseded'",
                old_id=old_function_id
            )
            
            # Create SUPERSEDES relationship from old to new
            self.add_edge(old_function_id, new_function_id, EdgeType.SUPERSEDES)
            
            logger.debug(f"Updated function version: {old_function_id} -> {new_function_id}")
            return new_function_id
    
    def get_latest_function_version(self, function_name: str, file_path: str = None) -> dict:
        """
        Traverses SUPERSEDES edges to find the most recent version of a function.
        
        Args:
            function_name (str): Name of the function to find
            file_path (str, optional): File path to narrow down search
            
        Returns:
            dict: Properties of the latest function version, or None if not found
        """
        with self.driver.session() as session:
            # Build the query based on whether file_path is provided
            if file_path:
                # Search for function by name and file path
                query = """
                MATCH (f:Function)
                WHERE f.properties.name = $function_name 
                  AND f.properties.file_path = $file_path
                  AND (f.properties.status IS NULL OR f.properties.status = 'active')
                
                // Check if this function is superseded by following SUPERSEDES edges
                OPTIONAL MATCH path = (f)-[:SUPERSEDES*]->(latest:Function)
                WHERE latest.properties.status = 'active' OR latest.properties.status IS NULL
                
                // Return the latest version if it exists, otherwise the original
                WITH f, latest, path
                ORDER BY length(path) DESC
                LIMIT 1
                
                RETURN COALESCE(latest.properties, f.properties) AS function_data,
                       COALESCE(id(latest), id(f)) AS function_id
                """
                result = session.run(query, function_name=function_name, file_path=file_path)
            else:
                # Search for function by name only
                query = """
                MATCH (f:Function)
                WHERE f.properties.name = $function_name
                  AND (f.properties.status IS NULL OR f.properties.status = 'active')
                
                // Check if this function is superseded by following SUPERSEDES edges
                OPTIONAL MATCH path = (f)-[:SUPERSEDES*]->(latest:Function)
                WHERE latest.properties.status = 'active' OR latest.properties.status IS NULL
                
                // Return the latest version if it exists, otherwise the original
                WITH f, latest, path
                ORDER BY length(path) DESC
                LIMIT 1
                
                RETURN COALESCE(latest.properties, f.properties) AS function_data,
                       COALESCE(id(latest), id(f)) AS function_id
                """
                result = session.run(query, function_name=function_name)
            
            record = result.single()
            if record:
                function_data = record["function_data"]
                function_data["node_id"] = record["function_id"]
                logger.debug(f"Found latest version of function '{function_name}': {function_data.get('node_id')}")
                return function_data
            else:
                logger.debug(f"No function found with name '{function_name}'" + (f" in file '{file_path}'" if file_path else ""))
                return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics for the graph store.
        
        Returns:
            Dictionary with performance metrics
        """
        cache_stats = self.cache_manager.get_stats()
        query_stats = self.performance_tracker.get_performance_summary()
        
        # Get materialized view stats
        popular_views = self.materialized_views.get_popular_views()
        
        return {
            'cache': cache_stats,
            'queries': query_stats,
            'materialized_views': {
                'total_views': len(self.materialized_views.materialized_views),
                'popular_views': popular_views
            },
            'optimization_candidates': self.performance_tracker.identify_optimization_candidates()
        }
    
    def warm_cache_for_common_patterns(self, common_file_patterns: List[str] = None) -> None:
        """
        Pre-warm cache with common query patterns.
        
        Args:
            common_file_patterns: Common file patterns to pre-load
        """
        if common_file_patterns is None:
            common_file_patterns = ['*.py', 'test_*.py', '*_test.py', 'main.py', '__init__.py']
        
        logger.info("Warming cache with common patterns...")
        
        for pattern in common_file_patterns:
            try:
                # Create mock plan for common patterns
                plan = {
                    'files_to_inspect': [pattern],
                    'functions_to_analyze': ['main', 'init', 'setup', 'run'],
                    'approach': 'common_pattern_warming'
                }
                
                # Cache result for future queries
                self.get_subgraph_by_plan(plan, use_compression=True)
                
            except Exception as e:
                logger.debug(f"Failed to warm cache for pattern {pattern}: {e}")
        
        logger.info("Cache warming completed")
    
    def optimize_for_task_patterns(self, task_history: List[Dict[str, Any]]) -> None:
        """
        Optimize the graph store based on historical task patterns.
        
        Args:
            task_history: List of previous task dictionaries
        """
        if not task_history:
            return
        
        logger.info(f"Optimizing graph store based on {len(task_history)} historical tasks")
        
        # Analyze common file patterns
        file_patterns = {}
        function_patterns = {}
        
        for task in task_history:
            # Count file patterns
            for file in task.get('files_to_inspect', []):
                file_patterns[file] = file_patterns.get(file, 0) + 1
            
            # Count function patterns  
            for func in task.get('functions_to_analyze', []):
                function_patterns[func] = function_patterns.get(func, 0) + 1
        
        # Get most common patterns
        common_files = sorted(file_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        common_functions = sorted(function_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Warm cache with common patterns
        common_file_list = [file for file, _ in common_files]
        self.warm_cache_for_common_patterns(common_file_list)
        
        logger.info("Graph store optimization completed")
    
    def cleanup_cache_and_views(self) -> None:
        """
        Clean up cache and materialized views to free memory.
        """
        logger.info("Cleaning up cache and materialized views")
        
        # Clear old cache entries
        self.cache_manager.clear()
        
        # Clear materialized views
        self.materialized_views.materialized_views.clear()
        
        logger.info("Cleanup completed")
    
    def _apply_context_compression(self, context: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, str]:
        """
        Apply hierarchical context compression to reduce token usage.
        
        Args:
            context: Raw context dictionary
            plan: Task plan for relevance scoring
            
        Returns:
            Compressed context sections
        """
        # Convert context to ContextItem objects
        context_items = []
        
        # Process functions
        for func_data in context.get('functions', []):
            if isinstance(func_data, dict) and func_data.get('name'):
                item = ContextItem(
                    content=func_data.get('code', func_data.get('body', '')),
                    content_type='function',
                    importance_score=self._calculate_importance_score(func_data, plan),
                    file_path=func_data.get('file_path', func_data.get('location', '')),
                    name=func_data.get('name', 'unknown'),
                    dependencies=set(func_data.get('calls', [])),
                    size_tokens=0  # Will be calculated automatically
                )
                context_items.append(item)
        
        # Process classes
        for class_data in context.get('classes', []):
            if isinstance(class_data, dict) and class_data.get('name'):
                item = ContextItem(
                    content=class_data.get('code', class_data.get('body', '')),
                    content_type='class',
                    importance_score=self._calculate_importance_score(class_data, plan),
                    file_path=class_data.get('file_path', class_data.get('location', '')),
                    name=class_data.get('name', 'unknown'),
                    dependencies=set(class_data.get('inherits_from', [])),
                    size_tokens=0
                )
                context_items.append(item)
        
        # Process related functions with lower importance
        for func_data in context.get('related_functions', []):
            if isinstance(func_data, dict) and func_data.get('name'):
                item = ContextItem(
                    content=func_data.get('code', func_data.get('body', '')),
                    content_type='function',
                    importance_score=self._calculate_importance_score(func_data, plan) * 0.7,  # Lower importance
                    file_path=func_data.get('file_path', func_data.get('location', '')),
                    name=func_data.get('name', 'unknown'),
                    dependencies=set(func_data.get('calls', [])),
                    size_tokens=0
                )
                context_items.append(item)
        
        if not context_items:
            return {}
        
        # Use adaptive context selector to determine optimal compression
        task_query = plan.get('approach', '') + ' ' + ' '.join(plan.get('files_to_inspect', []))
        adjusted_budget, compression_level = self.context_selector.select_compression_strategy(
            task_query, context_items, self.context_compressor.max_tokens
        )
        
        # Apply compression
        self.context_compressor.max_tokens = adjusted_budget
        compressed_sections = self.context_compressor.compress_context(context_items, task_query)
        
        return compressed_sections
    
    def _calculate_importance_score(self, item_data: Dict[str, Any], plan: Dict[str, Any]) -> float:
        """
        Calculate importance score for a context item.
        
        Args:
            item_data: Item data dictionary
            plan: Task plan
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        item_name = item_data.get('name', '')
        file_path = item_data.get('file_path', item_data.get('location', ''))
        
        # Boost score if item is in target files
        if file_path and any(target_file in file_path for target_file in plan.get('files_to_inspect', [])):
            score += 0.3
        
        # Boost score if item is in target functions
        if item_name and any(target_func in item_name for target_func in plan.get('functions_to_analyze', [])):
            score += 0.4
        
        # Boost score for items with error handling (useful for debugging)
        content = item_data.get('code', item_data.get('body', ''))
        if content:
            if any(keyword in content.lower() for keyword in ['except', 'raise', 'error', 'fail']):
                score += 0.1
            
            # Boost score for main/entry functions
            if any(keyword in item_name.lower() for keyword in ['main', 'run', 'execute', 'process']):
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0