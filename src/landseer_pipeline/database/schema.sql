-- =====================================================
-- Landseer Pipeline Results Database Schema
-- Version: 1.0
-- Created: 2026-01-09
-- =====================================================
-- This schema is designed for efficient querying of pipeline runs,
-- tool evaluations, and security metrics analysis.
-- =====================================================

-- Drop existing tables if they exist (in reverse dependency order)
DROP TABLE IF EXISTS output_file_provenance;
DROP TABLE IF EXISTS artifact_files;
DROP TABLE IF EXISTS artifact_nodes;
DROP TABLE IF EXISTS tool_executions;
DROP TABLE IF EXISTS combination_metrics;
DROP TABLE IF EXISTS combination_tools;
DROP TABLE IF EXISTS combinations;
DROP TABLE IF EXISTS pipeline_attacks;
DROP TABLE IF EXISTS pipeline_runs;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS datasets;
DROP TABLE IF EXISTS tools;
DROP TABLE IF EXISTS tool_categories;

-- =====================================================
-- REFERENCE TABLES (Lookup/Dimension Tables)
-- =====================================================

-- Tool categories/types for classification
CREATE TABLE tool_categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- Insert default categories
INSERT INTO tool_categories (category_name, description) VALUES
    ('adversarial', 'Adversarial robustness defenses'),
    ('outlier', 'Outlier/anomaly detection and removal'),
    ('differential_privacy', 'Differential privacy mechanisms'),
    ('watermarking', 'Model watermarking techniques'),
    ('fingerprinting', 'Model fingerprinting techniques'),
    ('compression', 'Model compression/pruning'),
    ('noop', 'No-operation placeholder'),
    ('unknown', 'Uncategorized tools');

-- Registry of all available tools
CREATE TABLE tools (
    tool_id INT AUTO_INCREMENT PRIMARY KEY,
    tool_name VARCHAR(100) NOT NULL UNIQUE,
    category_id INT,
    stage ENUM('pre_training', 'during_training', 'post_training', 'deployment') NOT NULL,
    container_image VARCHAR(255),
    container_command VARCHAR(500),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    FOREIGN KEY (category_id) REFERENCES tool_categories(category_id),
    INDEX idx_tool_stage (stage),
    INDEX idx_tool_category (category_id)
) ENGINE=InnoDB;

-- Dataset registry
CREATE TABLE datasets (
    dataset_id INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    variant VARCHAR(50) NOT NULL DEFAULT 'clean',
    version VARCHAR(50),
    num_classes INT,
    num_train_samples INT,
    num_test_samples INT,
    input_shape VARCHAR(50),
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE KEY uk_dataset (dataset_name, variant, version),
    INDEX idx_dataset_name (dataset_name),
    INDEX idx_dataset_variant (variant)
) ENGINE=InnoDB;

-- Model configurations registry
CREATE TABLE models (
    model_id INT AUTO_INCREMENT PRIMARY KEY,
    script_path VARCHAR(500) NOT NULL,
    content_hash CHAR(64) NOT NULL UNIQUE,
    framework ENUM('pytorch', 'tensorflow', 'onnx', 'other') DEFAULT 'pytorch',
    architecture_name VARCHAR(100),
    num_parameters BIGINT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_model_hash (content_hash),
    INDEX idx_model_framework (framework)
) ENGINE=InnoDB;

-- =====================================================
-- PIPELINE RUN TABLES
-- =====================================================

-- Main pipeline runs table
CREATE TABLE pipeline_runs (
    run_id INT AUTO_INCREMENT PRIMARY KEY,
    pipeline_id CHAR(16) NOT NULL,
    run_timestamp DATETIME NOT NULL,
    config_file_path VARCHAR(500),
    attack_config_path VARCHAR(500),
    config_hash CHAR(64),
    
    -- Foreign keys to dimension tables
    dataset_id INT,
    model_id INT,
    
    -- Run metadata
    total_combinations INT DEFAULT 0,
    successful_combinations INT DEFAULT 0,
    failed_combinations INT DEFAULT 0,
    total_duration_sec DECIMAL(12,3),
    
    -- Status tracking
    status ENUM('running', 'completed', 'failed', 'partial') DEFAULT 'running',
    error_message TEXT,
    
    -- Git/versioning info
    git_commit VARCHAR(40),
    landseer_version VARCHAR(20),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id),
    FOREIGN KEY (model_id) REFERENCES models(model_id),
    
    UNIQUE KEY uk_pipeline_run (pipeline_id, run_timestamp),
    INDEX idx_pipeline_id (pipeline_id),
    INDEX idx_run_timestamp (run_timestamp),
    INDEX idx_run_status (status),
    INDEX idx_dataset (dataset_id)
) ENGINE=InnoDB;

-- Attack configuration for each pipeline run
CREATE TABLE pipeline_attacks (
    attack_id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    
    -- Attack types enabled
    backdoor_enabled BOOLEAN DEFAULT FALSE,
    adversarial_enabled BOOLEAN DEFAULT FALSE,
    outlier_enabled BOOLEAN DEFAULT FALSE,
    carlini_enabled BOOLEAN DEFAULT FALSE,
    watermarking_enabled BOOLEAN DEFAULT FALSE,
    fingerprinting_enabled BOOLEAN DEFAULT FALSE,
    inference_enabled BOOLEAN DEFAULT FALSE,
    other_enabled BOOLEAN DEFAULT FALSE,
    
    -- Attack parameters (JSON for flexibility)
    attack_params JSON,
    
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    INDEX idx_attack_run (run_id)
) ENGINE=InnoDB;

-- =====================================================
-- COMBINATION TABLES
-- =====================================================

-- Pipeline combinations (each unique tool combination)
CREATE TABLE combinations (
    combination_id INT AUTO_INCREMENT PRIMARY KEY,
    run_id INT NOT NULL,
    combination_code VARCHAR(20) NOT NULL,  -- e.g., 'comb_010'
    combination_index INT NOT NULL,
    
    -- Execution metadata
    status ENUM('pending', 'running', 'success', 'failure', 'skipped') DEFAULT 'pending',
    error_message TEXT,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    total_duration_sec DECIMAL(10,3),
    
    -- Output path
    output_directory VARCHAR(500),
    
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
    
    UNIQUE KEY uk_combination (run_id, combination_code),
    INDEX idx_comb_run (run_id),
    INDEX idx_comb_status (status),
    INDEX idx_comb_code (combination_code)
) ENGINE=InnoDB;

-- Tools used in each combination (many-to-many relationship)
CREATE TABLE combination_tools (
    id INT AUTO_INCREMENT PRIMARY KEY,
    combination_id INT NOT NULL,
    tool_id INT NOT NULL,
    stage ENUM('pre_training', 'during_training', 'post_training', 'deployment') NOT NULL,
    tool_order INT NOT NULL DEFAULT 0,  -- Order within stage
    
    FOREIGN KEY (combination_id) REFERENCES combinations(combination_id) ON DELETE CASCADE,
    FOREIGN KEY (tool_id) REFERENCES tools(tool_id),
    
    UNIQUE KEY uk_comb_tool_stage (combination_id, tool_id, stage, tool_order),
    INDEX idx_comb_tool (combination_id),
    INDEX idx_tool (tool_id),
    INDEX idx_stage (stage)
) ENGINE=InnoDB;

-- Evaluation metrics for each combination
CREATE TABLE combination_metrics (
    metric_id INT AUTO_INCREMENT PRIMARY KEY,
    combination_id INT NOT NULL UNIQUE,
    
    -- Accuracy metrics
    acc_train_clean DECIMAL(6,4),
    acc_test_clean DECIMAL(6,4),
    
    -- Adversarial robustness metrics
    pgd_accuracy DECIMAL(6,4),
    carlini_l2_accuracy DECIMAL(6,4),
    fgsm_accuracy DECIMAL(6,4),
    autoattack_accuracy DECIMAL(6,4),
    
    -- Out-of-distribution detection
    ood_auc DECIMAL(6,4),
    ood_fpr_at_95_tpr DECIMAL(6,4),
    
    -- Model fingerprinting
    fingerprinting_score DECIMAL(6,4),
    
    -- Backdoor attack metrics
    attack_success_rate DECIMAL(6,4),
    clean_accuracy_after_attack DECIMAL(6,4),
    
    -- Privacy metrics
    privacy_epsilon DECIMAL(10,4),
    dp_accuracy DECIMAL(6,4),
    mia_auc DECIMAL(6,4),  -- Membership Inference Attack
    eps_estimate DECIMAL(10,4),
    
    -- Watermarking metrics
    watermark_accuracy DECIMAL(6,4),
    watermark_detection_rate DECIMAL(6,4),
    
    -- Model efficiency metrics
    model_size_mb DECIMAL(10,3),
    inference_time_ms DECIMAL(10,3),
    flops BIGINT,
    
    -- Timestamps
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (combination_id) REFERENCES combinations(combination_id) ON DELETE CASCADE,
    
    -- Indexes for common query patterns
    INDEX idx_acc_test (acc_test_clean),
    INDEX idx_pgd (pgd_accuracy),
    INDEX idx_carlini (carlini_l2_accuracy),
    INDEX idx_ood (ood_auc),
    INDEX idx_mia (mia_auc),
    INDEX idx_watermark (watermark_accuracy)
) ENGINE=InnoDB;

-- =====================================================
-- TOOL EXECUTION TABLES
-- =====================================================

-- Individual tool execution records
CREATE TABLE tool_executions (
    execution_id INT AUTO_INCREMENT PRIMARY KEY,
    combination_id INT NOT NULL,
    tool_id INT NOT NULL,
    stage ENUM('pre_training', 'during_training', 'post_training', 'deployment') NOT NULL,
    
    -- Cache information
    cache_key CHAR(64),
    cache_hit BOOLEAN DEFAULT FALSE,
    
    -- Execution details
    status ENUM('pending', 'running', 'success', 'failure', 'cached') DEFAULT 'pending',
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    duration_sec DECIMAL(10,3),
    
    -- Resource usage
    peak_memory_mb DECIMAL(10,3),
    gpu_memory_mb DECIMAL(10,3),
    cpu_percent DECIMAL(5,2),
    
    -- Output information
    output_path VARCHAR(500),
    log_path VARCHAR(500),
    error_message TEXT,
    
    -- Container info
    container_id VARCHAR(64),
    exit_code INT,
    
    FOREIGN KEY (combination_id) REFERENCES combinations(combination_id) ON DELETE CASCADE,
    FOREIGN KEY (tool_id) REFERENCES tools(tool_id),
    
    INDEX idx_exec_comb (combination_id),
    INDEX idx_exec_tool (tool_id),
    INDEX idx_exec_stage (stage),
    INDEX idx_exec_status (status),
    INDEX idx_exec_cache_key (cache_key),
    INDEX idx_exec_cache_hit (cache_hit)
) ENGINE=InnoDB;

-- =====================================================
-- ARTIFACT TABLES (Content-Addressable Storage)
-- =====================================================

-- Artifact nodes (content-addressable)
CREATE TABLE artifact_nodes (
    node_id INT AUTO_INCREMENT PRIMARY KEY,
    node_hash CHAR(64) NOT NULL UNIQUE,
    tool_identity_hash CHAR(64),
    tool_name VARCHAR(100),
    stage ENUM('pre_training', 'during_training', 'post_training', 'deployment'),
    
    -- Parent nodes (stored as JSON array of hashes)
    parent_hashes JSON,
    
    -- Execution info
    duration_sec DECIMAL(10,3),
    total_size_bytes BIGINT,
    
    -- Storage path
    storage_path VARCHAR(500),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    access_count INT DEFAULT 1,
    
    INDEX idx_node_hash (node_hash),
    INDEX idx_tool_identity (tool_identity_hash),
    INDEX idx_tool_name (tool_name),
    INDEX idx_node_stage (stage)
) ENGINE=InnoDB;

-- Files within artifact nodes
CREATE TABLE artifact_files (
    file_id INT AUTO_INCREMENT PRIMARY KEY,
    node_id INT NOT NULL,
    relative_path VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT,
    file_hash CHAR(64),
    mime_type VARCHAR(100),
    
    FOREIGN KEY (node_id) REFERENCES artifact_nodes(node_id) ON DELETE CASCADE,
    
    UNIQUE KEY uk_node_file (node_id, relative_path),
    INDEX idx_file_node (node_id),
    INDEX idx_file_hash (file_hash)
) ENGINE=InnoDB;

-- Output file provenance (tracks which tool produced each output)
CREATE TABLE output_file_provenance (
    provenance_id INT AUTO_INCREMENT PRIMARY KEY,
    combination_id INT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    source_path VARCHAR(500),
    stage ENUM('pre_training', 'during_training', 'post_training', 'deployment'),
    tool_name VARCHAR(100),
    was_copied BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (combination_id) REFERENCES combinations(combination_id) ON DELETE CASCADE,
    
    UNIQUE KEY uk_comb_file (combination_id, file_name),
    INDEX idx_prov_comb (combination_id),
    INDEX idx_prov_tool (tool_name)
) ENGINE=InnoDB;

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- View: Complete combination results with all metrics
CREATE OR REPLACE VIEW v_combination_results AS
SELECT 
    pr.pipeline_id,
    pr.run_timestamp,
    pr.config_file_path,
    d.dataset_name,
    d.variant AS dataset_variant,
    c.combination_code,
    c.status AS combination_status,
    c.total_duration_sec,
    cm.acc_train_clean,
    cm.acc_test_clean,
    cm.pgd_accuracy,
    cm.carlini_l2_accuracy,
    cm.ood_auc,
    cm.fingerprinting_score,
    cm.attack_success_rate,
    cm.privacy_epsilon,
    cm.dp_accuracy,
    cm.watermark_accuracy,
    cm.mia_auc,
    cm.eps_estimate
FROM pipeline_runs pr
JOIN datasets d ON pr.dataset_id = d.dataset_id
JOIN combinations c ON pr.run_id = c.run_id
LEFT JOIN combination_metrics cm ON c.combination_id = cm.combination_id;

-- View: Tool combination summary (shows tools used per combination)
CREATE OR REPLACE VIEW v_combination_tools_summary AS
SELECT 
    c.combination_id,
    c.combination_code,
    pr.pipeline_id,
    GROUP_CONCAT(CASE WHEN ct.stage = 'pre_training' THEN t.tool_name END ORDER BY ct.tool_order SEPARATOR ', ') AS pre_training_tools,
    GROUP_CONCAT(CASE WHEN ct.stage = 'during_training' THEN t.tool_name END ORDER BY ct.tool_order SEPARATOR ', ') AS during_training_tools,
    GROUP_CONCAT(CASE WHEN ct.stage = 'post_training' THEN t.tool_name END ORDER BY ct.tool_order SEPARATOR ', ') AS post_training_tools,
    GROUP_CONCAT(CASE WHEN ct.stage = 'deployment' THEN t.tool_name END ORDER BY ct.tool_order SEPARATOR ', ') AS deployment_tools
FROM combinations c
JOIN pipeline_runs pr ON c.run_id = pr.run_id
LEFT JOIN combination_tools ct ON c.combination_id = ct.combination_id
LEFT JOIN tools t ON ct.tool_id = t.tool_id
GROUP BY c.combination_id, c.combination_code, pr.pipeline_id;

-- View: Tool execution performance
CREATE OR REPLACE VIEW v_tool_performance AS
SELECT 
    t.tool_name,
    t.stage,
    tc.category_name,
    COUNT(*) AS total_executions,
    SUM(CASE WHEN te.status = 'success' THEN 1 ELSE 0 END) AS successful_executions,
    SUM(CASE WHEN te.cache_hit THEN 1 ELSE 0 END) AS cache_hits,
    AVG(te.duration_sec) AS avg_duration_sec,
    MIN(te.duration_sec) AS min_duration_sec,
    MAX(te.duration_sec) AS max_duration_sec,
    AVG(te.peak_memory_mb) AS avg_peak_memory_mb
FROM tools t
LEFT JOIN tool_categories tc ON t.category_id = tc.category_id
LEFT JOIN tool_executions te ON t.tool_id = te.tool_id
GROUP BY t.tool_id, t.tool_name, t.stage, tc.category_name;

-- View: Pipeline run summary
CREATE OR REPLACE VIEW v_pipeline_summary AS
SELECT 
    pr.run_id,
    pr.pipeline_id,
    pr.run_timestamp,
    pr.status,
    d.dataset_name,
    d.variant,
    m.architecture_name,
    pr.total_combinations,
    pr.successful_combinations,
    pr.failed_combinations,
    ROUND(pr.successful_combinations * 100.0 / NULLIF(pr.total_combinations, 0), 2) AS success_rate_pct,
    pr.total_duration_sec,
    pa.adversarial_enabled,
    pa.backdoor_enabled,
    pa.watermarking_enabled
FROM pipeline_runs pr
LEFT JOIN datasets d ON pr.dataset_id = d.dataset_id
LEFT JOIN models m ON pr.model_id = m.model_id
LEFT JOIN pipeline_attacks pa ON pr.run_id = pa.run_id;

-- View: Best performing combinations by accuracy
CREATE OR REPLACE VIEW v_best_combinations AS
SELECT 
    pr.pipeline_id,
    c.combination_code,
    d.dataset_name,
    cm.acc_test_clean,
    cm.pgd_accuracy,
    cm.carlini_l2_accuracy,
    cm.ood_auc,
    -- Composite robustness score (weighted average)
    ROUND(
        (COALESCE(cm.acc_test_clean, 0) * 0.3 + 
         COALESCE(cm.pgd_accuracy, 0) * 0.25 + 
         COALESCE(cm.carlini_l2_accuracy, 0) * 0.25 + 
         COALESCE(cm.ood_auc, 0) * 0.2), 4
    ) AS robustness_score
FROM combinations c
JOIN pipeline_runs pr ON c.run_id = pr.run_id
JOIN datasets d ON pr.dataset_id = d.dataset_id
JOIN combination_metrics cm ON c.combination_id = cm.combination_id
WHERE c.status = 'success'
ORDER BY robustness_score DESC;

-- View: Cache efficiency analysis
CREATE OR REPLACE VIEW v_cache_efficiency AS
SELECT 
    pr.pipeline_id,
    pr.run_timestamp,
    COUNT(*) AS total_tool_executions,
    SUM(CASE WHEN te.cache_hit THEN 1 ELSE 0 END) AS cache_hits,
    SUM(CASE WHEN NOT te.cache_hit THEN 1 ELSE 0 END) AS cache_misses,
    ROUND(SUM(CASE WHEN te.cache_hit THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS cache_hit_rate_pct,
    SUM(CASE WHEN NOT te.cache_hit THEN te.duration_sec ELSE 0 END) AS compute_time_sec,
    SUM(CASE WHEN te.cache_hit THEN te.duration_sec ELSE 0 END) AS saved_time_sec
FROM pipeline_runs pr
JOIN combinations c ON pr.run_id = c.run_id
JOIN tool_executions te ON c.combination_id = te.combination_id
GROUP BY pr.pipeline_id, pr.run_timestamp;

-- =====================================================
-- STORED PROCEDURES FOR COMMON OPERATIONS
-- =====================================================

DELIMITER //

-- Procedure: Get combinations with specific tool
CREATE PROCEDURE sp_get_combinations_with_tool(
    IN p_tool_name VARCHAR(100),
    IN p_stage VARCHAR(20)
)
BEGIN
    SELECT 
        pr.pipeline_id,
        c.combination_code,
        cm.acc_test_clean,
        cm.pgd_accuracy,
        c.status
    FROM combinations c
    JOIN pipeline_runs pr ON c.run_id = pr.run_id
    JOIN combination_tools ct ON c.combination_id = ct.combination_id
    JOIN tools t ON ct.tool_id = t.tool_id
    LEFT JOIN combination_metrics cm ON c.combination_id = cm.combination_id
    WHERE t.tool_name = p_tool_name
      AND (p_stage IS NULL OR ct.stage = p_stage);
END //

-- Procedure: Compare tool effectiveness
CREATE PROCEDURE sp_compare_tool_effectiveness(
    IN p_tool_name_1 VARCHAR(100),
    IN p_tool_name_2 VARCHAR(100),
    IN p_dataset_name VARCHAR(100)
)
BEGIN
    SELECT 
        t.tool_name,
        COUNT(*) AS num_combinations,
        AVG(cm.acc_test_clean) AS avg_clean_accuracy,
        AVG(cm.pgd_accuracy) AS avg_pgd_accuracy,
        AVG(cm.carlini_l2_accuracy) AS avg_carlini_accuracy,
        AVG(cm.ood_auc) AS avg_ood_auc
    FROM tools t
    JOIN combination_tools ct ON t.tool_id = ct.tool_id
    JOIN combinations c ON ct.combination_id = c.combination_id
    JOIN pipeline_runs pr ON c.run_id = pr.run_id
    JOIN datasets d ON pr.dataset_id = d.dataset_id
    JOIN combination_metrics cm ON c.combination_id = cm.combination_id
    WHERE t.tool_name IN (p_tool_name_1, p_tool_name_2)
      AND (p_dataset_name IS NULL OR d.dataset_name = p_dataset_name)
      AND c.status = 'success'
    GROUP BY t.tool_name;
END //

-- Procedure: Get pipeline run details
CREATE PROCEDURE sp_get_pipeline_details(
    IN p_pipeline_id CHAR(16)
)
BEGIN
    -- Run summary
    SELECT * FROM v_pipeline_summary WHERE pipeline_id = p_pipeline_id;
    
    -- Combination results
    SELECT * FROM v_combination_results WHERE pipeline_id = p_pipeline_id;
    
    -- Tool performance for this run
    SELECT 
        te.stage,
        t.tool_name,
        COUNT(*) AS executions,
        SUM(CASE WHEN te.cache_hit THEN 1 ELSE 0 END) AS cache_hits,
        AVG(te.duration_sec) AS avg_duration
    FROM tool_executions te
    JOIN tools t ON te.tool_id = t.tool_id
    JOIN combinations c ON te.combination_id = c.combination_id
    JOIN pipeline_runs pr ON c.run_id = pr.run_id
    WHERE pr.pipeline_id = p_pipeline_id
    GROUP BY te.stage, t.tool_name
    ORDER BY te.stage, t.tool_name;
END //

-- Procedure: Find best tool combination for a metric
CREATE PROCEDURE sp_find_best_combination(
    IN p_dataset_name VARCHAR(100),
    IN p_metric VARCHAR(50),
    IN p_limit INT
)
BEGIN
    SET @sql = CONCAT(
        'SELECT c.combination_code, d.dataset_name, cm.', p_metric, ' AS metric_value,
         vcts.pre_training_tools, vcts.during_training_tools, 
         vcts.post_training_tools, vcts.deployment_tools
         FROM combinations c
         JOIN pipeline_runs pr ON c.run_id = pr.run_id
         JOIN datasets d ON pr.dataset_id = d.dataset_id
         JOIN combination_metrics cm ON c.combination_id = cm.combination_id
         JOIN v_combination_tools_summary vcts ON c.combination_id = vcts.combination_id
         WHERE d.dataset_name = ? AND c.status = ''success'' AND cm.', p_metric, ' IS NOT NULL
         ORDER BY cm.', p_metric, ' DESC
         LIMIT ?'
    );
    
    PREPARE stmt FROM @sql;
    SET @dataset = p_dataset_name;
    SET @lim = p_limit;
    EXECUTE stmt USING @dataset, @lim;
    DEALLOCATE PREPARE stmt;
END //

-- Procedure: Insert or update tool
CREATE PROCEDURE sp_upsert_tool(
    IN p_tool_name VARCHAR(100),
    IN p_category_name VARCHAR(50),
    IN p_stage VARCHAR(20),
    IN p_container_image VARCHAR(255),
    IN p_container_command VARCHAR(500)
)
BEGIN
    DECLARE v_category_id INT;
    
    -- Get or create category
    SELECT category_id INTO v_category_id 
    FROM tool_categories WHERE category_name = p_category_name;
    
    IF v_category_id IS NULL THEN
        INSERT INTO tool_categories (category_name) VALUES (p_category_name);
        SET v_category_id = LAST_INSERT_ID();
    END IF;
    
    -- Upsert tool
    INSERT INTO tools (tool_name, category_id, stage, container_image, container_command)
    VALUES (p_tool_name, v_category_id, p_stage, p_container_image, p_container_command)
    ON DUPLICATE KEY UPDATE
        category_id = v_category_id,
        container_image = p_container_image,
        container_command = p_container_command,
        updated_at = CURRENT_TIMESTAMP;
END //

DELIMITER ;

-- =====================================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =====================================================

-- Composite indexes for common query patterns
CREATE INDEX idx_comb_run_status ON combinations(run_id, status);
CREATE INDEX idx_metrics_accuracy ON combination_metrics(acc_test_clean, pgd_accuracy);
CREATE INDEX idx_exec_comb_stage ON tool_executions(combination_id, stage);
CREATE INDEX idx_run_dataset_status ON pipeline_runs(dataset_id, status);

-- =====================================================
-- SAMPLE DATA INSERTION (for testing)
-- =====================================================

-- Insert common datasets
INSERT INTO datasets (dataset_name, variant, num_classes, num_train_samples, num_test_samples, input_shape) VALUES
    ('cifar10', 'clean', 10, 50000, 10000, '32x32x3'),
    ('cifar10', 'poisoned', 10, 50000, 10000, '32x32x3'),
    ('cifar100', 'clean', 100, 50000, 10000, '32x32x3'),
    ('mnist', 'clean', 10, 60000, 10000, '28x28x1'),
    ('imagenet', 'clean', 1000, 1281167, 50000, '224x224x3');

-- Insert common tools
INSERT INTO tools (tool_name, category_id, stage, container_image, description) VALUES
    ('noop', 7, 'pre_training', NULL, 'No-operation passthrough for pre-training'),
    ('pre-xgbod', 2, 'pre_training', 'ghcr.io/landseer-project/pre_xgbod:v2', 'XGBoost-based outlier detection'),
    ('in-trades', 1, 'during_training', 'ghcr.io/landseer-project/in_trades:v2', 'TRADES adversarial training'),
    ('in-pgd', 1, 'during_training', 'ghcr.io/landseer-project/in_pgd:v2', 'PGD adversarial training'),
    ('in-free', 1, 'during_training', 'ghcr.io/landseer-project/in_free:v2', 'Free adversarial training'),
    ('post_noop', 7, 'post_training', NULL, 'No-operation passthrough for post-training'),
    ('fine_pruning', 6, 'post_training', 'ghcr.io/landseer-project/fine_pruning:v2', 'Fine pruning for model compression'),
    ('deploy_noop', 7, 'deployment', NULL, 'No-operation passthrough for deployment'),
    ('deploy_dp', 3, 'deployment', 'ghcr.io/landseer-project/deploy_dp:v2', 'Differential privacy deployment wrapper'),
    ('post_magnet', 1, 'deployment', 'ghcr.io/landseer-project/magnet:v2', 'MagNet adversarial defense');
