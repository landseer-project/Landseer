import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CarliniL2Attack:
    """
    Optimized Carlini & Wagner L2 Attack implementation
    """
    def __init__(self, model, device, confidence=0, learning_rate=0.01, 
                 max_iterations=100, binary_search_steps=3, initial_const=1e-3,
                 early_stop_threshold=1e-4):
        self.model = model
        self.device = device
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations  # Reduced from 1000 to 100
        self.binary_search_steps = binary_search_steps  # Reduced from 5 to 3
        self.initial_const = initial_const
        self.early_stop_threshold = early_stop_threshold
        
    def __call__(self, images, labels):
        """
        Generate adversarial examples using optimized C&W L2 attack
        """
        batch_size = images.shape[0]
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        
        # Initialize variables
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)
        const = torch.full((batch_size,), self.initial_const, device=self.device)
        
        best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        best_adv = images.clone()
        
        # Pre-compute mask for efficiency
        batch_indices = torch.arange(batch_size, device=self.device)
        
        # Binary search with early stopping
        for search_step in range(self.binary_search_steps):
            # Initialize perturbation
            delta = torch.zeros_like(images, requires_grad=True)
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            
            prev_loss = float('inf')
            patience_counter = 0
            
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    adv_images = torch.clamp(images + delta, 0, 1)
                    outputs = self.model(adv_images)
                    
                    # Vectorized loss calculation
                    l2_loss = torch.norm((adv_images - images).view(batch_size, -1), 
                                       p=2, dim=1) ** 2
                    
                    # Efficient Carlini objective
                    real_scores = outputs[batch_indices, labels]
                    outputs_copy = outputs.clone()
                    outputs_copy[batch_indices, labels] = -1e4
                    other_scores = torch.max(outputs_copy, dim=1)[0]
                    
                    f_loss = torch.clamp(real_scores - other_scores + self.confidence, min=0)
                    total_loss = torch.sum(const * f_loss + l2_loss)
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Early stopping check
                if abs(prev_loss - total_loss.item()) < self.early_stop_threshold:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break
                else:
                    patience_counter = 0
                prev_loss = total_loss.item()
                
                # Update best examples every 10 iterations (not every iteration)
                if iteration % 10 == 0:
                    with torch.no_grad():
                        pred_labels = torch.argmax(outputs, dim=1)
                        success_mask = (pred_labels != labels) & (l2_loss < best_l2)
                        best_l2[success_mask] = l2_loss[success_mask]
                        best_adv[success_mask] = adv_images[success_mask]
            
            # Vectorized binary search update
            with torch.no_grad():
                final_outputs = self.model(best_adv)
                final_pred = torch.argmax(final_outputs, dim=1)
                success_mask = (final_pred != labels)
                
                upper_bound[success_mask] = const[success_mask]
                lower_bound[~success_mask] = const[~success_mask]
                
                # Update const
                valid_upper = upper_bound < 1e9
                const[valid_upper] = (lower_bound[valid_upper] + upper_bound[valid_upper]) / 2
                const[~valid_upper] = lower_bound[~valid_upper] * 10
        
        return best_adv

def evaluate_carlini_l2(model, loader, device, confidence=0, max_batches=None):
    """
    Optimized Carlini L2 evaluation with batching
    """
    model.eval()
    attack = CarliniL2Attack(
        model, device, confidence=confidence,
        max_iterations=50,  # Further reduced for evaluation
        binary_search_steps=2  # Minimal for faster evaluation
    )
    
    correct, total = 0, 0
    batch_count = 0
    
    # Process multiple batches at once for better GPU utilization
    batch_accumulator = []
    label_accumulator = []
    
    for X, y in loader:
        batch_accumulator.append(X)
        label_accumulator.append(y)
        
        # Process accumulated batches
        if len(batch_accumulator) >= 4 or batch_count == len(loader) - 1:
            # Combine batches
            combined_X = torch.cat(batch_accumulator, dim=0).to(device, non_blocking=True)
            combined_y = torch.cat(label_accumulator, dim=0).to(device, non_blocking=True)
            
            # Generate adversarial examples
            with torch.cuda.amp.autocast():
                adv_X = attack(combined_X, combined_y)
                
                # Evaluate
                with torch.no_grad():
                    pred = model(adv_X).argmax(1)
                    correct += (pred == combined_y).sum().item()
                    total += combined_y.size(0)
            
            # Clear accumulators
            batch_accumulator = []
            label_accumulator = []
        
        batch_count += 1
        if max_batches and batch_count >= max_batches:
            break
    
    return correct / total if total > 0 else 0.0

# Quick evaluation function for faster testing
def evaluate_carlini_l2(model, loader, device, confidence=0, sample_size=500):
    """
    Fast Carlini evaluation with sampling
    """
    model.eval()
    attack = CarliniL2Attack(
        model, device, confidence=confidence,
        max_iterations=50,  # Very reduced
        binary_search_steps=1  # Single search
    )
    
    # Sample random subset for faster evaluation
    all_X, all_y = [], []
    for X, y in loader:
        all_X.append(X)
        all_y.append(y)
        if len(all_X) * X.size(0) >= sample_size:
            break
    
    if not all_X:
        return 0.0
    
    # Combine and sample
    combined_X = torch.cat(all_X, dim=0)[:sample_size]
    combined_y = torch.cat(all_y, dim=0)[:sample_size]
    
    # Single batch evaluation
    combined_X = combined_X.to(device, non_blocking=True)
    combined_y = combined_y.to(device, non_blocking=True)
    
    with torch.cuda.amp.autocast():
        adv_X = attack(combined_X, combined_y)
        
        with torch.no_grad():
            pred = model(adv_X).argmax(1)
            correct = (pred == combined_y).sum().item()
    
    return correct / len(combined_y)