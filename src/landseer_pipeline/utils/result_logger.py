from pathlib import Path
import csv

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
                f.write("combination,pre_training,in_training,post_training,deployment,dataset_name,dataset_type,acc_train_clean,acc_test_clean,acc_robust,ood_auc,fingerprinting,asr,privacy_epsilon,dp_accuracy,total_duration\n")

    def log_tool(self, combination, stage, tool_name, cache_key, output_path, duration, status):
        with open(self.tools_csv, "a") as f:
            f.write(f"{self.pipeline_id},{combination},{stage},{tool_name},{cache_key},{duration:.2f},{status},{output_path}\n")
    
    def log_combination(self, combination, tools_by_stage, dataset_name, dataset_type, acc, duration):
        def extract_names(tools):
            return [tool.name for tool in tools] if tools else []
        
        pre_training = extract_names(tools_by_stage.get("pre_training", []))
        in_training = extract_names(tools_by_stage.get("during_training", []))
        post_training = extract_names(tools_by_stage.get("post_training", []))
        deployment = extract_names(tools_by_stage.get("deployment", []))
        acc_train_clean = acc.get("clean_train_accuracy", -1)
        acc_clean = acc.get("clean_test_accuracy", -1)
        acc_robust = acc.get("robust_accuracy", -1)
        ood_auc = acc.get("ood_auc", -1)
        fingerprinting_acc = acc.get("fingerprinting", -1)
        asr = acc.get("backdoor_asr", -1)
        privacy_epsilon = acc.get("privacy_epsilon", -1)
        dp_acc = acc.get("dp_accuracy", -1)
         
        file_exists = self.combinations_csv.exists()
        with open(self.combinations_csv, "a", newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            if not file_exists:
                writer.writerow([
                "combination", "pre_training", "in_training", "post_training", "deployment",
                "dataset_name", "dataset_type", "acc_train_clean", "acc_test_clean", "acc_robust",
                "ood_auc", "fingerprinting", "asr", "privacy_epsilon", "dp_accuracy", "total_duration"
                ])
            writer.writerow([
            combination, str(pre_training), str(in_training), str(post_training), str(deployment),
            dataset_name, dataset_type, round(acc_train_clean, 4), round(acc_clean, 4),
            round(acc_robust, 4), round(ood_auc, 4), round(fingerprinting_acc, 4),
            round(asr, 4), str(privacy_epsilon), str(dp_acc), round(duration, 2)
            ])
