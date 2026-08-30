"""Create a Gitee release for taichi-matrix v0.1.0"""
import requests
import os
import json

USERNAME = 'sun-yongji-yuyubenyuan_admin'
PASSWORD = '898ab3d52116b55dd02515df914c23d2'
OWNER = 'sun-yongji-yuyubenyuan_admin'
REPO = 'taichi-matrix'
TAG = 'v0.1.0'

# Step 1: Create the release
url = f'https://gitee.com/api/v5/repos/{OWNER}/{REPO}/releases'
body_text = """太极矩阵 TaiChi Matrix v0.1.0 正式发布

包含M1-M6六大模块，159项测试全通过

### 安装
```
pip install git+https://gitee.com/sun-yongji-yuyubenyuan_admin/taichi-matrix.git
```

### 离线安装
下载 .whl 文件后：
```
pip install taichi_matrix-0.1.0-py3-none-any.whl
```

### 六大模块一览
| 模块 | 测试 | 核心指标 |
|------|------|---------|
| M1 Router 路由引擎 | 26/26 | C6群论三模式路由，熵平衡 1.47 |
| M2 MTP 多令牌预测 | 34/34 | 六头深度调度，湍流耦合 100:1 |
| M3 Quant 量化器 | 28/28 | 4.3x 压缩 / 87.3% 保真度 |
| M4 HexAttn 六边形注意力 | 26/26 | 对角线 2.56x，head 多样性 1.0 |
| M5 Correct 误差校正 | 28/28 | 噪声降 69.7%，置信度 98% |
| M6 Integrate 集成测试 | 17/17 | 端到端 0.79ms |
| **总计** | **159/159** | **全通过** |

### 许可
Apache-2.0
"""

resp = requests.post(url, auth=(USERNAME, PASSWORD), json={
    'tag_name': TAG,
    'target_commitish': 'master',
    'name': '太极矩阵 TaiChi Matrix v0.1.0',
    'body': body_text,
    'prerelease': False,
})

print(f'Create release: HTTP {resp.status_code}')
if resp.status_code == 201:
    release = resp.json()
    release_id = release['id']
    print(f'Release created! ID={release_id}')
    print(f'URL: {release["html_url"]}')
    
    # Step 2: Upload assets
    dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dist')
    upload_url = f'https://gitee.com/api/v5/repos/{OWNER}/{REPO}/releases/{release_id}/attach_files'
    
    for fname in ['taichi_matrix-0.1.0-py3-none-any.whl', 'taichi_matrix-0.1.0.tar.gz']:
        fpath = os.path.join(dist_dir, fname)
        if not os.path.exists(fpath):
            print(f'  SKIP {fname}: not found')
            continue
        with open(fpath, 'rb') as f:
            files = {'file': (fname, f, 'application/octet-stream')}
            resp2 = requests.post(upload_url, auth=(USERNAME, PASSWORD), files=files)
            print(f'  Upload {fname}: HTTP {resp2.status_code}')
            if resp2.status_code != 201:
                print(f'    Error: {resp2.text[:200]}')
            else:
                print(f'    OK: {resp2.json().get("download_url", "")}')
else:
    print(f'Error: {resp.text[:500]}')
