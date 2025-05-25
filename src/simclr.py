import torch
import torch.nn as nn
import torch.nn.functional as F
from src.text_embedding import TextEmbeddingModel

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, in_dim, out_dim):
        super(ClassificationHead, self).__init__()
        self.dense1 = nn.Linear(in_dim, in_dim//4)
        self.dense2 = nn.Linear(in_dim//4, in_dim//16)
        self.out_proj = nn.Linear(in_dim//16, out_dim)

        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.dense1.bias, std=1e-6)
        nn.init.normal_(self.dense2.bias, std=1e-6)
        nn.init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, features):
        x = features
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x

class SimCLR_Classifier_SCL(nn.Module):
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_SCL, self).__init__()
        
        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        self.device=self.model.model.device
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.esp=torch.tensor(1e-6,device=self.device)
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        
        self.a=torch.tensor(opt.a,device=self.device)
        self.d=torch.tensor(opt.d,device=self.device)
        self.only_classifier=opt.only_classifier


    def get_encoder(self):
        return self.model

    def _compute_logits(self, q,q_index1, q_index2,q_label,k,k_index1,k_index2,k_label):
        def cosine_similarity_matrix(q, k):

            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        
        logits=cosine_similarity_matrix(q,k)/self.temperature

        q_labels=q_label.view(-1, 1)# N,1
        k_labels=k_label.view(1, -1)# 1,N+K

        same_label=(q_labels==k_labels)# N,N+K

        #model:model set
        pos_logits_model = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_model=logits*torch.logical_not(same_label)
        logits_model=torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1) 

        return logits_model
    
    def forward(self, batch, indices1, indices2,label):
        bsz = batch['input_ids'].size(0)
        q = self.model(batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index1 = self.fabric.all_gather(indices1).view(-1)
        k_index2 = self.fabric.all_gather(indices2).view(-1)
        #q:N
        #k:4N
        logits_label = self._compute_logits(q,indices1, indices2,label,k,k_index1,k_index2,k_label)
        
        out = self.classifier(q)
        
        if self.opt.AA:
            loss_classfiy = F.cross_entropy(out, indices1)
        else:
            loss_classfiy = F.cross_entropy(out, label)

        gt = torch.zeros(bsz, dtype=torch.long,device=logits_label.device)

        if self.only_classifier:
            loss_label = torch.tensor(0,device=self.device)
        else:
            loss_label = F.cross_entropy(logits_label, gt)

        loss = self.a*loss_label+self.d*loss_classfiy
        if self.training:
            return loss,loss_label,loss_classfiy,k,k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            return loss,out,k,k_label


class SimCLR_Classifier_test(nn.Module):
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier_test, self).__init__()
        
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        self.device=self.model.model.device
    
    def forward(self, batch):
        q = self.model(batch)
        out = self.classifier(q)
        return out

class SimCLR_Classifier(nn.Module):
    def __init__(self, opt,fabric):
        super(SimCLR_Classifier, self).__init__()

        self.temperature = opt.temperature
        self.opt=opt
        self.fabric = fabric

        self.model = TextEmbeddingModel(opt.model_name)
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.model.device)
            self.model.load_state_dict(state_dict)
  
        self.device=self.model.model.device
        self.esp=torch.tensor(1e-6,device=self.device)
        self.a = torch.tensor(opt.a,device=self.device)
        self.b = torch.tensor(opt.b,device=self.device)
        self.c = torch.tensor(opt.c,device=self.device)
        self.d = torch.tensor(opt.d,device=self.device)

        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        self.only_classifier = opt.only_classifier


    def get_encoder(self):
        return self.model

    # computing loss for model
    def _compute_logits(self, q,q_index1, q_index2,q_label,k,k_index1,k_index2,k_label):
        def cosine_similarity_matrix(q, k):

            q_norm = F.normalize(q,dim=-1)
            k_norm = F.normalize(k,dim=-1)
            cosine_similarity = q_norm@k_norm.T
            
            return cosine_similarity
        
        logits=cosine_similarity_matrix(q,k)/self.temperature

        q_index1=q_index1.view(-1, 1)# N,1
        q_index2=q_index2.view(-1, 1)# N,1
        q_labels=q_label.view(-1, 1)# N,1

        k_index1=k_index1.view(1, -1)# 1,N+K
        k_index2=k_index2.view(1, -1)
        k_labels=k_label.view(1, -1)# 1,N+K

        same_model=(q_index1==k_index1)
        same_set=(q_index2==k_index2)# N,N+K
        same_label=(q_labels==k_labels)# N,N+K

        is_human=(q_label==1).view(-1)
        is_machine=(q_label==0).view(-1)

        pos_logits_human = torch.sum(logits*same_label,dim=1)/torch.max(torch.sum(same_label,dim=1),self.esp)
        neg_logits_human=logits*torch.logical_not(same_label)
        logits_human=torch.cat((pos_logits_human.unsqueeze(1), neg_logits_human), dim=1)
        logits_human=logits_human[is_human]

        #model:model set
        pos_logits_model = torch.sum(logits*same_model,dim=1)/torch.max(torch.sum(same_model,dim=1),self.esp)# N
        neg_logits_model=logits*torch.logical_not(same_model)# N,N+K
        logits_model=torch.cat((pos_logits_model.unsqueeze(1), neg_logits_model), dim=1)
        logits_model=logits_model[is_machine]
        #model set:label
        pos_logits_set = torch.sum(logits*torch.logical_xor(same_set,same_model),dim=1)/torch.max(torch.sum(torch.logical_xor(same_set,same_model),dim=1),self.esp)
        neg_logits_set=logits*torch.logical_not(same_set)
        logits_set=torch.cat((pos_logits_set.unsqueeze(1), neg_logits_set), dim=1)
        logits_set=logits_set[is_machine]      
        #label:label
        pos_logits_label = torch.sum(logits*torch.logical_xor(same_set,same_label),dim=1)/torch.max(torch.sum(torch.logical_xor(same_set,same_label),dim=1),self.esp)
        neg_logits_label=logits*torch.logical_not(same_label)
        logits_label=torch.cat((pos_logits_label.unsqueeze(1), neg_logits_label), dim=1)
        logits_label=logits_label[is_machine]        

        return logits_model,logits_set,logits_label,logits_human
    
    def forward(self, encoded_batch, indices1, indices2,label):
        # print(len(text))
        q = self.model(encoded_batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index1 = self.fabric.all_gather(indices1).view(-1)
        k_index2 = self.fabric.all_gather(indices2).view(-1)
        #q:N
        #k:4N
        logits_model,logits_set,logits_label,logits_human = self._compute_logits(q,indices1, indices2,label,k,k_index1,k_index2,k_label)
        out = self.classifier(q)
        
        if self.opt.AA:
            loss_classfiy = F.cross_entropy(out, indices1)
        else:
            loss_classfiy = F.cross_entropy(out, label)

        gt_model = torch.zeros(logits_model.size(0), dtype=torch.long,device=logits_model.device)
        gt_set = torch.zeros(logits_set.size(0), dtype=torch.long,device=logits_set.device)
        gt_label = torch.zeros(logits_label.size(0), dtype=torch.long,device=logits_label.device)
        gt_human = torch.zeros(logits_human.size(0), dtype=torch.long,device=logits_human.device)


        loss_model =  F.cross_entropy(logits_model, gt_model)
        loss_set = F.cross_entropy(logits_set, gt_set)
        loss_label = F.cross_entropy(logits_label, gt_label)
        if logits_human.numel()!=0:
            loss_human = F.cross_entropy(logits_human.to(torch.float64), gt_human)
        else:
            loss_human=torch.tensor(0,device=self.device)

        loss = self.a*loss_model + self.b*loss_set + self.c*loss_label+(self.a+self.b+self.c)*loss_human+self.d*loss_classfiy
        if self.training:
            if self.opt.AA:
                return loss,loss_model,loss_set,loss_label,loss_human,loss_classfiy,k,k_index1
            else:
                return loss,loss_model,loss_set,loss_label,loss_classfiy,loss_human,k,k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            if self.opt.AA:
                return loss,out,k,k_index1
            else:
                return loss,out,k,k_label

class SIGNature_Classifier(nn.Module):
    """
    SIGNature: Sigmoid Pairwise Learning for AI-Generated Text Detection
    Replaces InfoNCE loss with sigmoid pairwise loss inspired by SigLIP
    """
    def __init__(self, opt, fabric):
        super(SIGNature_Classifier, self).__init__()

        self.temperature = opt.temperature
        self.opt = opt
        self.fabric = fabric

        self.model = TextEmbeddingModel(opt.model_name)
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.model.device)
            self.model.load_state_dict(state_dict)
  
        self.device = self.model.model.device
        self.esp = torch.tensor(1e-6, device=self.device)
        
        # Loss weights for different levels
        self.a = torch.tensor(opt.a, device=self.device)  # model level
        self.b = torch.tensor(opt.b, device=self.device)  # set level  
        self.c = torch.tensor(opt.c, device=self.device)  # label level
        self.d = torch.tensor(opt.d, device=self.device)  # classification

        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        self.only_classifier = opt.only_classifier

    def get_encoder(self):
        return self.model

    def _compute_sigmoid_loss(self, q, q_index1, q_index2, q_label, k, k_index1, k_index2, k_label):
        """
        Compute sigmoid pairwise loss as described in SIGNature paper
        Lsig(i) = -1/|K+i| Σ log σ(sij) - 1/|K-i| Σ log(1-σ(sik))
        """
        def cosine_similarity_matrix(q, k):
            q_norm = F.normalize(q, dim=-1)
            k_norm = F.normalize(k, dim=-1)
            cosine_similarity = q_norm @ k_norm.T
            return cosine_similarity
        
        # Compute similarities scaled by temperature
        similarities = cosine_similarity_matrix(q, k) / self.temperature  # N x K
        
        # Reshape for broadcasting
        q_index1 = q_index1.view(-1, 1)  # N, 1
        q_index2 = q_index2.view(-1, 1)  # N, 1
        q_labels = q_label.view(-1, 1)   # N, 1

        k_index1 = k_index1.view(1, -1)  # 1, K
        k_index2 = k_index2.view(1, -1)  # 1, K
        k_labels = k_label.view(1, -1)   # 1, K

        # Define positive/negative pairs for different levels
        same_model = (q_index1 == k_index1)  # exact same model
        same_set = (q_index2 == k_index2)    # same model family/set
        same_label = (q_labels == k_labels)  # same human/AI label

        is_human = (q_label == 1).view(-1)
        is_machine = (q_label == 0).view(-1)

        total_loss = 0.0
        loss_components = {}

        # Model-level loss (for AI texts only)
        if torch.any(is_machine):
            machine_similarities = similarities[is_machine]  # M x K
            machine_same_model = same_model[is_machine]      # M x K
            
            # Positive pairs: same model
            pos_mask = machine_same_model
            # Negative pairs: different model
            neg_mask = ~machine_same_model
            
            loss_model = self._sigmoid_pairwise_loss(machine_similarities, pos_mask, neg_mask)
            loss_components['model'] = loss_model
            total_loss += self.a * loss_model
        else:
            loss_components['model'] = torch.tensor(0.0, device=self.device)

        # Set-level loss (for AI texts only) 
        if torch.any(is_machine):
            machine_similarities = similarities[is_machine]
            machine_same_set = same_set[is_machine]
            machine_same_model = same_model[is_machine]
            
            # Positive pairs: same set but different model (XOR operation)
            pos_mask = machine_same_set & ~machine_same_model
            # Negative pairs: different set
            neg_mask = ~machine_same_set
            
            loss_set = self._sigmoid_pairwise_loss(machine_similarities, pos_mask, neg_mask)
            loss_components['set'] = loss_set
            total_loss += self.b * loss_set
        else:
            loss_components['set'] = torch.tensor(0.0, device=self.device)

        # Label-level loss (for AI texts only)
        if torch.any(is_machine):
            machine_similarities = similarities[is_machine]
            machine_same_label = same_label[is_machine]
            machine_same_set = same_set[is_machine]
            
            # Positive pairs: same label but different set (XOR operation) 
            pos_mask = machine_same_label & ~machine_same_set
            # Negative pairs: different label
            neg_mask = ~machine_same_label
            
            loss_label = self._sigmoid_pairwise_loss(machine_similarities, pos_mask, neg_mask)
            loss_components['label'] = loss_label
            total_loss += self.c * loss_label
        else:
            loss_components['label'] = torch.tensor(0.0, device=self.device)

        # Human-level loss (for human texts only)
        if torch.any(is_human):
            human_similarities = similarities[is_human]
            human_same_label = same_label[is_human]
            
            # Positive pairs: same label (human)
            pos_mask = human_same_label
            # Negative pairs: different label (AI)
            neg_mask = ~human_same_label
            
            loss_human = self._sigmoid_pairwise_loss(human_similarities, pos_mask, neg_mask)
            loss_components['human'] = loss_human
            total_loss += (self.a + self.b + self.c) * loss_human
        else:
            loss_components['human'] = torch.tensor(0.0, device=self.device)

        return total_loss, loss_components

    def _sigmoid_pairwise_loss(self, similarities, pos_mask, neg_mask):
        """
        Compute sigmoid pairwise loss for given similarities and masks
        """
        # Apply sigmoid to similarities
        probs = torch.sigmoid(similarities)
        
        # Positive loss: -log(σ(sij)) for positive pairs
        pos_loss = 0.0
        if torch.any(pos_mask):
            pos_probs = probs * pos_mask.float()
            pos_count = torch.sum(pos_mask.float(), dim=1, keepdim=True)
            pos_count = torch.clamp(pos_count, min=1)  # avoid division by zero
            pos_loss = -torch.sum(torch.log(pos_probs + self.esp), dim=1) / pos_count.squeeze()
            pos_loss = torch.mean(pos_loss)

        # Negative loss: -log(1-σ(sik)) for negative pairs  
        neg_loss = 0.0
        if torch.any(neg_mask):
            neg_probs = probs * neg_mask.float()
            neg_count = torch.sum(neg_mask.float(), dim=1, keepdim=True)
            neg_count = torch.clamp(neg_count, min=1)  # avoid division by zero
            neg_loss = -torch.sum(torch.log(1 - neg_probs + self.esp), dim=1) / neg_count.squeeze()
            neg_loss = torch.mean(neg_loss)

        return pos_loss + neg_loss

    def forward(self, encoded_batch, indices1, indices2, label):
        q = self.model(encoded_batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        k_index1 = self.fabric.all_gather(indices1).view(-1)
        k_index2 = self.fabric.all_gather(indices2).view(-1)

        # Compute sigmoid pairwise loss instead of InfoNCE
        if self.only_classifier:
            sigmoid_loss = torch.tensor(0.0, device=self.device)
            loss_components = {
                'model': torch.tensor(0.0, device=self.device),
                'set': torch.tensor(0.0, device=self.device),
                'label': torch.tensor(0.0, device=self.device),
                'human': torch.tensor(0.0, device=self.device)
            }
        else:
            sigmoid_loss, loss_components = self._compute_sigmoid_loss(
                q, indices1, indices2, label, k, k_index1, k_index2, k_label
            )

        # Classification loss
        out = self.classifier(q)
        if self.opt.AA:
            loss_classify = F.cross_entropy(out, indices1)
        else:
            loss_classify = F.cross_entropy(out, label)

        # Total loss: sigmoid contrastive + classification
        total_loss = sigmoid_loss + self.d * loss_classify

        if self.training:
            if self.opt.AA:
                return total_loss, loss_components['model'], loss_components['set'], loss_components['label'], loss_classify, loss_components['human'], k, k_index1
            else:
                return total_loss, loss_components['model'], loss_components['set'], loss_components['label'], loss_classify, loss_components['human'], k, k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            if self.opt.AA:
                return total_loss, out, k, k_index1
            else:
                return total_loss, out, k, k_label


class SIGNature_Classifier_SCL(nn.Module):
    """
    SIGNature single contrastive loss version (simplified)
    Similar to SimCLR_Classifier_SCL but with sigmoid pairwise loss
    """
    def __init__(self, opt, fabric):
        super(SIGNature_Classifier_SCL, self).__init__()
        
        self.temperature = opt.temperature
        self.opt = opt
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        self.device = self.model.model.device
        
        if opt.resum:
            state_dict = torch.load(opt.pth_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        self.esp = torch.tensor(1e-6, device=self.device)
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        
        self.a = torch.tensor(opt.a, device=self.device)
        self.d = torch.tensor(opt.d, device=self.device)
        self.only_classifier = opt.only_classifier

    def get_encoder(self):
        return self.model

    def _compute_sigmoid_loss_simple(self, q, q_label, k, k_label):
        """
        Simplified sigmoid loss for single-level contrastive learning
        """
        def cosine_similarity_matrix(q, k):
            q_norm = F.normalize(q, dim=-1)
            k_norm = F.normalize(k, dim=-1)
            cosine_similarity = q_norm @ k_norm.T
            return cosine_similarity
        
        similarities = cosine_similarity_matrix(q, k) / self.temperature
        
        q_labels = q_label.view(-1, 1)  # N, 1
        k_labels = k_label.view(1, -1)  # 1, K
        
        same_label = (q_labels == k_labels)  # N, K
        
        # Apply sigmoid
        probs = torch.sigmoid(similarities)
        
        # Positive pairs: same label
        pos_mask = same_label.float()
        pos_count = torch.sum(pos_mask, dim=1, keepdim=True)
        pos_count = torch.clamp(pos_count, min=1)
        pos_loss = -torch.sum(torch.log(probs + self.esp) * pos_mask, dim=1) / pos_count.squeeze()
        
        # Negative pairs: different label
        neg_mask = (~same_label).float()
        neg_count = torch.sum(neg_mask, dim=1, keepdim=True)
        neg_count = torch.clamp(neg_count, min=1)
        neg_loss = -torch.sum(torch.log(1 - probs + self.esp) * neg_mask, dim=1) / neg_count.squeeze()
        
        return torch.mean(pos_loss + neg_loss)

    def forward(self, batch, indices1, indices2, label):
        bsz = batch['input_ids'].size(0)
        q = self.model(batch)
        k = q.clone().detach()
        k = self.fabric.all_gather(k).view(-1, k.size(1))
        k_label = self.fabric.all_gather(label).view(-1)
        
        # Sigmoid contrastive loss
        if self.only_classifier:
            loss_label = torch.tensor(0, device=self.device)
        else:
            loss_label = self._compute_sigmoid_loss_simple(q, label, k, k_label)
        
        # Classification loss
        out = self.classifier(q)
        if self.opt.AA:
            loss_classify = F.cross_entropy(out, indices1)
        else:
            loss_classify = F.cross_entropy(out, label)

        loss = self.a * loss_label + self.d * loss_classify
        
        if self.training:
            return loss, loss_label, loss_classify, k, k_label
        else:
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            return loss, out, k, k_label

class SIGNature_Classifier_test(nn.Module):
    """
    SIGNature model for inference/testing only
    Similar to SimCLR_Classifier_test but compatible with SIGNature training
    """
    def __init__(self, opt, fabric):
        super(SIGNature_Classifier_test, self).__init__()
        
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)
        self.device = self.model.model.device
    
    def forward(self, batch):
        q = self.model(batch)
        out = self.classifier(q)
        return out
