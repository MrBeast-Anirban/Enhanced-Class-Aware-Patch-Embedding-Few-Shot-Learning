import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class CPEA(nn.Module):
    def __init__(self, in_dim=384):
        super(CPEA, self).__init__()

        self.fc1 = Mlp(in_features=in_dim, hidden_features=int(in_dim/4), out_features=in_dim)
        self.fc_norm1 = nn.LayerNorm(in_dim)

        self.fc2 = Mlp(in_features=197**2,  hidden_features=256, out_features=1)
        self.fc3 = Mlp(in_features = 384, hidden_features = 128, out_features = 384)
        self.fc4 = Mlp(in_features = 384, hidden_features = 128, out_features = 5)

    def forward(self, feat_query, feat_shot, args):
        _, n, c = feat_query.size()
        k_shot = args.shot  # number of shots per class
        num_classes = args.way  # number of ways
        
        feat_query = self.fc1(torch.mean(feat_query, dim=1, keepdim=True)) + feat_query
        feat_shot  = self.fc1(torch.mean(feat_shot, dim=1, keepdim=True)) + feat_shot

        feat_query = self.fc_norm1(feat_query)
        feat_shot  = self.fc_norm1(feat_shot)

        query_class = feat_query[:, 0, :].unsqueeze(1)
        query_image = feat_query[:, 1:, :]
        support_class = feat_shot[:, 0, :].unsqueeze(1)
        support_image = feat_shot[:, 1:, :]

        feat_query = query_image + 2.0 * query_class
        feat_shot = support_image + 2.0 * support_class

        feat_query_mean = torch.mean(feat_query, dim = 1, keepdim = True)
        feat_shot_mean = torch.mean(feat_shot, dim = 1, keepdim = True)

        # Compute new features for query and shot
        feat_query_new = self.fc3(feat_query_mean)
        feat_shot_new = self.fc3(feat_shot_mean)
        
        # Calculate number of examples for reshaping
        num_query = feat_query.size(0)  # number of queries
        num_shot = feat_shot.size(0)    # number of shots

        # Ensure reshaping matches the actual batch sizes
        feat_query_new = feat_query_new.view(num_query, 1, c)
        feat_shot_new = feat_shot_new.view(num_shot, 1, c)

        feat_query = torch.cat((feat_query, feat_query_new), dim=1)
        feat_shot = torch.cat((feat_shot, feat_shot_new), dim=1)
        
    
        feat_query_final = self.fc4(feat_query_new) #(75, 5)
        feat_shot_final = self.fc4(feat_shot_new) #(25, 5) for 5-shot, (5, 5) for 1-shot
    
        
        feat_query = F.normalize(feat_query, p=2, dim=2)
        feat_query = feat_query - torch.mean(feat_query, dim=2, keepdim=True)
        
        feat_shot = feat_shot.contiguous().reshape(k_shot, -1, n, c)
        feat_shot = feat_shot.mean(dim=0)
        feat_shot = F.normalize(feat_shot, p=2, dim=2)
        feat_shot = feat_shot - torch.mean(feat_shot, dim=2, keepdim=True)

        results = []
        for idx in range(feat_query.size(0)):
            tmp_query = feat_query[idx]
            tmp_query = tmp_query.unsqueeze(0)
            out = torch.matmul(feat_shot, tmp_query.transpose(1, 2))
            out = out.flatten(1)
            out = self.fc2(out.pow(2))
            out = out.transpose(0, 1)
            results.append(out)

        results2 = torch.cat((feat_query_final, feat_shot_final), dim = 0).view(feat_shot.shape[0] + feat_query.shape[0], -1)
 

        return results, results2













"""
       # 25x196x384, 75x196x384
       # mean dim=1, 25x1x384, 75x1x384
       feat_query_mean = torch.mean(feat_query, dim = 1, keepdim = True)
       feat_shot_mean = torch.mean(feat_shot, dim = 1, keepdim = True)
       # fc in=384,hidden=?,out=384
       feat_query_new = self.fc3(feat_query_mean).view(75, 1, 384)
       feat_shot_new = self.fc3(feat_shot_mean).view(25, 1, 384)
       # 1. concat  25x1x384 , 25x196x384 .. same for query --new feat_shot and new feat query   
       feat_query = torch.cat((feat_query, feat_query_new), dim = 1)
       feat_shot = torch.cat((feat_shot, feat_shot_new), dim = 1)       
       # 2. pass 25x1x384, 75x1x384 to an fc in =384, out=5, hidden=?
       feat_query_final = self.fc4(feat_query_new)
       # feat_shot_final = self.fc4(feat_shot_new)
"""