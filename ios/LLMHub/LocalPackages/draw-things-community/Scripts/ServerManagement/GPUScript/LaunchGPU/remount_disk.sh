# 1. Create the new mount point directories
sudo mkdir -p /mnt/models /mnt/loraModels

# 2. Unmount the current disks
sudo umount /mnt/disk1
sudo umount /mnt/disk2

# 3. Mount to the new locations
sudo mount /dev/nvme1n1p1 /mnt/models
sudo mount /dev/nvme2n1p1 /mnt/loraModels

# 4. Verify the new mounts
df -h | grep mnt


# sudo nano /etc/fstab
# ```

# Find the lines for `/mnt/disk1` and `/mnt/disk2` and change them to:
# ```
# /dev/nvme1n1p1  /mnt/models      ext4  defaults  0  2
# /dev/nvme2n1p1  /mnt/loraModels  ext4  defaults  0  2


#!/bin/bash
# MergerFS 存储合并脚本
# 用途：将 /mnt/models 和 /mnt/loraModels/models_extra 合并到 /mnt/official_models

set -e

echo "=== 1. 安装 mergerfs ==="
apt update && apt install -y mergerfs

echo "=== 2. 创建目录 ==="
mkdir -p /mnt/loraModels/models_extra
mkdir -p /mnt/official_models

echo "=== 3. 挂载 mergerfs ==="
# 只读模式（默认）
# mergerfs -o ro,use_ino,cache.files=partial,dropcacheonclose=true,category.search=ff \
#     /mnt/models:/mnt/loraModels/models_extra \
#     /mnt/official_models

# 可写模式（新文件优先写到空间多的盘）
mergerfs -o defaults,allow_other,use_ino,cache.files=partial,dropcacheonclose=true,category.create=epmfs \
    /mnt/models:/mnt/loraModels/models_extra \
    /mnt/official_models

echo "=== 4. 验证 ==="
df -h /mnt/official_models
ls /mnt/official_models

echo "=== 5. 持久化到 fstab（可选）==="
FSTAB_LINE='/mnt/models:/mnt/loraModels/models_extra /mnt/official_models fuse.mergerfs defaults,allow_other,use_ino,cache.files=partial,dropcacheonclose=true,category.create=epmfs 0 0'

if ! grep -q "official_models" /etc/fstab; then
    echo "$FSTAB_LINE" >> /etc/fstab
    echo "已添加到 /etc/fstab"
else
    echo "fstab 中已存在相关配置，跳过"
fi

echo "=== 完成 ==="