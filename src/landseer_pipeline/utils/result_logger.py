from pathlib import Path

class ResultLogger:
    def __init__(self, results_dir, pipeline_id):
        self.pipeline_id = pipeline_id
        self.results_dir = Path(results_dir)
        self.tools_csv = self.results_dir / "results_tools.csv"
        self.combinations_csv = self.results_dir / "results_combinations.csv"
        self._ensure_headers()

    def _ensure_headers(self):
        if not self.tools_csv.exists():
            with open(self.tools_csv, "w") as f:
                f.write("pipeline_id,combination,stage,tool_name,cache_key,duration_sec,status,output_path\n")
        if not self.combinations_csv.exists():
            with open(self.combinations_csv, "w") as f:
                f.write("pipeline_id,combination,pre_training,in_training,post_training,dataset_name,dataset_type,acc_train_clean,acc_test_clean,acc_robust,ood_auc,fingerprinting,asr,privacy_epsilon,total_duration\n")

    def log_tool(self, combination, stage, tool_name, cache_key, output_path, duration, status):
        with open(self.tools_csv, "a") as f:
            f.write(f"{self.pipeline_id},{combination},{stage},{tool_name},{cache_key},{duration:.2f},{status},{output_path}\n")
    
    def log_combination(self, combination, tools_by_stage, dataset_name, dataset_type, acc, duration):
        def extract_names(tools):
            return [tool.name for tool in tools] if tools else []
        
        pre_training = extract_names(tools_by_stage.get("pre_training", []))
        in_training = extract_names(tools_by_stage.get("during_training", []))
        post_training = extract_names(tools_by_stage.get("post_training", []))

        acc_train_clean = acc.get("clean_train_accuracy", 0.0)
        acc_clean = acc.get("clean_test_accuracy", 0.0)
        acc_robust = acc.get("robust_accuracy", 0.0)
        ood_auc = acc.get("ood_auc", 0.0)
        fingerprinting_acc = acc.get("fingerprinting", 0.0)
        asr = acc.get("backdoor_asr", 0.0)
        privacy_epsilon = acc.get("privacy_epsilon", 0.0)
        
        with open(self.combinations_csv, "a") as f:
            f.write(f"{self.pipeline_id},{combination},{pre_training},{in_training},{post_training},"
                f"{dataset_name},{dataset_type},{acc_train_clean:.4f},{acc_clean:.4f},{acc_robust:.4f},"
                f"{ood_auc:.4f},{fingerprinting_acc:.4f},{asr:.4f},{privacy_epsilon:.4f},{duration:.2f}\n")
 
