import torch

from model.abstract import PJFModel


class Pop(PJFModel):
    def __init__(self, config, pool):
        super(Pop, self).__init__(config, pool)

        self.item_cnt = torch.nn.Parameter(
            torch.zeros(pool.geek_num, 1, dtype=torch.long, device=self.device),
            requires_grad=False
        )

        self.max_cnt = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.long),
            requires_grad=False
        )
        self.loss = torch.nn.Parameter(torch.zeros(1))

    def calculate_loss(self, interaction):
        item = interaction['job_id']
        self.item_cnt[item, :] += 1
        self.max_cnt.data = torch.max(self.item_cnt, dim=0)[0]
        return self.loss

    def predict(self, interaction):
        item = interaction['job_id']
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        return result[item]
