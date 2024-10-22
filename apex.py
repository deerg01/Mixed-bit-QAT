import torch
from apex import amp

# 간단한 모델과 옵티마이저 정의
model = torch.nn.Linear(10, 10).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# APEX 초기화
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

print("APEX가 성공적으로 설치되었습니다.")
