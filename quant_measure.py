import pandas as pd
from statistics import mean
json_obj = pd.read_json('compare_restored_clean.json')

## PSNR Measures

psnr_all = list(json_obj.psnr)
psnr_avg_all = mean(frame['psnr_avg'] for frame in psnr_all)
mse_avg_all = mean(frame['mse_avg'] for frame in psnr_all)

## SSIM Measures

ssim_all = list(json_obj.ssim)
ssim_avg_all = mean(frame['ssim_avg'] for frame in ssim_all)
print(ssim_avg_all)

# print((list(json_obj.psnr)[:5]))
# print(json_obj.ssim[:5])


