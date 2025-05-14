_base_ = [
    'configs/_base_/models/resnet50.py',  # 基础模型结构配置
    'configs/_base_/datasets/imagenet_bs32.py',  # 数据集基础配置
    'configs/_base_/schedules/imagenet_bs256.py',  # 学习率调度等训练策略
    'configs/_base_/default_runtime.py',  # 默认运行环境配置
]

# 模型结构与预训练权重设置
model = dict(
    head=dict(num_classes=5, topk=(1,)),  # 分类头类别数和topk
    backbone=dict(
        frozen_stages=3,  # 冻结前3个stage参数
        init_cfg=dict(
            type='Pretrained',  # 使用预训练权重
            checkpoint='checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth',  # 权重文件路径
            prefix='backbone',
        )
    )
)

# 数据预处理参数
# 均值、方差与ImageNet一致，保证输入分布
# num_classes需与实际类别数一致
# to_rgb=True表示BGR转RGB
#
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    num_classes=5,
)

dataset_type = 'ImageNet'  # 数据集类型
# 数据集根目录（相对当前工作目录）
data_root = '../flower_dataset'
# 读取类别名称
classes = [c.strip() for c in open(f'{data_root}/classes.txt')]

# 训练集dataloader配置
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/train',  # 训练图片文件夹
        ann_file=f'{data_root}/train.txt',  # 训练标注文件
        classes=classes,  # 类别列表
        data_root=data_root,  # 根目录
        pipeline=[  # 数据增强与预处理流程
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ],
        split='train',
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),  # 打乱数据
    pin_memory=True,
)

# 验证集dataloader配置
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/val',  # 验证图片文件夹
        ann_file=f'{data_root}/val.txt',  # 验证标注文件
        classes=classes,
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ],
        split='val',
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),  # 不打乱
    pin_memory=True,
)

val_cfg = dict()
val_evaluator = dict(type='Accuracy', topk=(1,))  # 验证指标：top1准确率

# 测试集dataloader配置（与验证集一致）
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_prefix=f'{data_root}/val',
        ann_file=f'{data_root}/val.txt',
        classes=classes,
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ],
        split='val',
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
    pin_memory=True,
)

test_cfg = dict()
test_evaluator = dict(type='Accuracy', topk=(1,))  # 测试指标

# 优化器与学习率调度
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',  # AdamW优化器
        lr=0.001,  # 初始学习率
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        _delete_=True
    )
)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),  # 线性预热
    dict(type='ExponentialLR', gamma=0.9, by_epoch=True)  # 指数衰减
]
auto_scale_lr = dict(base_batch_size=256)  # 自动学习率缩放

# 训练策略
train_cfg = dict(
    by_epoch=True,  # 按epoch训练
    max_epochs=20,  # 最大epoch数
    val_interval=1,  # 每个epoch验证一次
)
