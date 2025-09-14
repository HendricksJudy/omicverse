# OmicVerse 单细胞分析平台

基于Web的单细胞数据分析平台，提供直观的可视化界面和强大的分析功能。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install flask flask_cors scanpy numpy pandas werkzeug
```

### 2. 启动服务器
```bash
python3 start_server.py
```

### 3. 访问平台
打开浏览器访问: `http://localhost:5050`

## 📁 项目结构

```
omicverse_web/
├── app.py                              # Flask后端应用
├── single_cell_analysis_standalone.html # 主页面
├── start_server.py                     # 服务器启动脚本
├── create_sample_data.py               # 示例数据生成
├── quick_start.sh                      # 快速启动脚本
└── static/                            # 静态资源
    ├── css/                           # 样式文件
    │   ├── bootstrap.min.css          # Bootstrap框架
    │   ├── vendors.min.css            # 第三方组件
    │   ├── theme.min.css              # 主题样式
    │   ├── dark-mode.css              # 黑夜模式
    │   ├── daterangepicker.min.css    # 日期选择器
    │   └── single-cell-fixed.css      # 单细胞分析样式
    ├── js/                            # JavaScript文件
    │   ├── vendors.min.js             # 第三方组件
    │   ├── common-init.min.js         # 通用初始化
    │   └── single-cell.js             # 单细胞分析逻辑
    ├── font/                          # 字体文件
    └── picture/                       # 图片资源
```

## 🎯 主要功能

- **数据上传**: 支持H5AD格式的单细胞数据文件
- **数据可视化**: UMAP、t-SNE等降维可视化
- **分析工具**: 预处理、聚类、差异分析等
- **主题切换**: 支持明暗主题切换
- **响应式设计**: 适配不同屏幕尺寸

## 🛠️ 技术栈

- **后端**: Flask + Python
- **前端**: HTML5 + CSS3 + JavaScript
- **可视化**: Plotly.js
- **图标**: Feather Icons + Font Awesome
- **样式**: Bootstrap 5

## 📝 使用说明

1. **上传数据**: 拖拽或点击上传H5AD文件
2. **选择分析**: 从左侧导航选择分析类型
3. **设置参数**: 在右侧面板调整分析参数
4. **运行分析**: 点击运行按钮开始分析
5. **查看结果**: 在可视化区域查看分析结果

## 🔧 开发说明

- 修改前端界面: 编辑 `single_cell_analysis_standalone.html`
- 修改后端逻辑: 编辑 `app.py`
- 修改样式: 编辑 `static/css/` 中的CSS文件
- 修改交互: 编辑 `static/js/single-cell.js`

## 📄 许可证

本项目基于MIT许可证开源。