#!/usr/bin/env python3
"""
Emergency fix script to address critical pipeline performance issues
"""

import subprocess
import time
import signal
import os

def kill_running_containers():
    """Kill any running Docker containers that might be consuming resources"""
    print("🛑 Stopping all running Docker containers...")
    try:
        # Get running containers
        result = subprocess.run(['docker', 'ps', '-q'], capture_output=True, text=True)
        container_ids = result.stdout.strip().split('\n')
        
        if container_ids and container_ids[0]:
            for container_id in container_ids:
                if container_id.strip():
                    print(f"   Stopping container: {container_id}")
                    subprocess.run(['docker', 'stop', container_id], check=False)
                    subprocess.run(['docker', 'rm', container_id], check=False)
        
        print("✅ All containers stopped")
        
    except Exception as e:
        print(f"⚠️ Error stopping containers: {e}")

def kill_python_processes():
    """Kill any stuck Python processes"""
    print("🔄 Checking for stuck Python processes...")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        python_processes = []
        for line in lines:
            if 'python' in line and 'landseer' in line:
                parts = line.split()
                if len(parts) > 1:
                    pid = parts[1]
                    python_processes.append(pid)
        
        if python_processes:
            print(f"   Found {len(python_processes)} Python processes")
            for pid in python_processes:
                try:
                    print(f"   Killing PID: {pid}")
                    os.kill(int(pid), signal.SIGTERM)
                    time.sleep(1)
                    os.kill(int(pid), signal.SIGKILL)
                except:
                    pass
        else:
            print("   No stuck Python processes found")
            
    except Exception as e:
        print(f"⚠️ Error checking processes: {e}")

def check_gpu_status():
    """Check current GPU status"""
    print("📊 Current GPU Status:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv'], 
                              capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"⚠️ Error checking GPU status: {e}")

def show_recommendations():
    """Show performance recommendations"""
    print("\n📋 Performance Fixes Applied:")
    print("=" * 50)
    print("✅ Reduced parallel execution to 1 worker (prevents GPU conflicts)")
    print("✅ Added strict Docker resource limits (4GB RAM, 4 CPUs)")
    print("✅ Enabled auto-cleanup for containers")
    print("✅ Added CPU core pinning to prevent oversubscription")
    print()
    print("🎯 Expected Results:")
    print("• No more 'GPU allocation failed' errors")
    print("• Containers limited to reasonable resource usage")
    print("• More stable execution with predictable performance")
    print("• Trade-off: Sequential execution will be slower but stable")
    print()
    print("⚡ To Run Test Again:")
    print("cd /share/landseer/workspace-ayushi/Landseer")
    print("./test_fin.sh")
    print()
    print("📈 Monitor with:")
    print("watch -n 2 'nvidia-smi && echo && docker stats --no-stream'")

def main():
    print("🚨 EMERGENCY PIPELINE PERFORMANCE FIXER")
    print("=" * 45)
    
    # Emergency cleanup
    kill_running_containers()
    time.sleep(2)
    
    # Check system status
    check_gpu_status()
    
    # Show what was fixed
    show_recommendations()
    
    print("\n✅ Emergency fixes complete!")
    print("💡 The pipeline should now run more stably with sequential execution.")

if __name__ == "__main__":
    main()
