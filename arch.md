在Telegram 中，使用chat id （int64）来标识你与一个个人对话或者群聊，每个对话中使用 messsage id (int64)  来标识一个消息。整个系统因为需要基于用户自定义的Github Repo 来运行Action，所以后端实际上使用Github Repo ID 来对应每一个群聊或者个人。我希望对于群聊和个人的 bot 可以统一后端，而不是分别实现。个人可以直接 /summarize /setting 而群聊需要 @ 这个bot /summarize
群聊中只有管理员可以修改 /setting 以避免错误修改



注意dispatcher 与bot 部分信息解耦，理论上未来可以直接适配别的bot，所以不应该将平台特定的id信息发送到dispatcher，dispatcher 直接使用arxiv id，github repo id，like_num，dis_like_num 等通用数据。


对于部署的设备，我是用Oracle Cloud 的免费额度部署，有静态的公网ip，4核 24G 内存 200G 硬盘，10G 对象存储（存jsonl绰绰有余），50G的MySQL，两个20G的oracle DB。我倾向于使用MySQL部署所有的数据库。用户的pat 在数据中存储需要加密，目前是简单使用密钥加密后保存在sqlite中，与其他数据独立管理。

数据库部分，我希望能保持在python中使用async 异步访问，以减少性能开销。

与Telegram 通讯部分，目前使用1s 一次的长轮询实现，很灵活，也不依赖固定的公网IP，方便开发，同时由于指令都需要使用github action，尤其是其中的LLM api调用耗时较长，整体需要3-5分钟，所以使用webhook带来的低延迟优势意义不是很大。



如果需要一些docker 服务，可以直接在服务器中使用 podman或者podman-compose部署（个人偏好）但是目前的基础功能似乎并不涉及到容器的使用。后续（不在本次开发计划内）我会使用qdrant 容器来部署向量数据库用于相似论文检索，或者直接使用Oracle DB中的AI search 功能来减少云服务器中的资源占用。



/recommend 指令已经基本实现（除了PAT检索，硬编码PAT运行成功）

reaction 的处理还没有实现，需要结合数据库

/summarize 指令可以复用大部分代码，但是 github action部分我应该可以在1天内搞定

/start 下需要弹出一个配置指南，这个由我准备好文档放在服务器里即可。

```Mermaid
flowchart TB
    A([/recommend 指令手动或每日定时触发])
    A --> B[后端 src/bot/tg.py recommend]
    
    %% 绑定检查流程
    B --> C{查询chat id绑定的<br/>Github Repo ID与PTA}:::todo
    C -->|存在| D[发送到 src/dispatcher.py]
    C -->|不存在| E[提示用户进入/setting 设置]:::warning
    
    %% Github Action处理
    D --> F[async触发github action recommend.yml/summarize.yml<br/>处理对象存储不匹配的论文]:::action
    F -->|运行失败| H(提示用户运行失败):::warning
    F -->|运行成功| G[下载artifact到tmp路径并解压]
    
    %% 渲染推送流程
    G --> I[src/render.py独立进程池<br/>jinjia2转文本]
    I --> J[src/bot/tg.py发送消息<br/>记录chat/message/arxiv id]:::todo
    J --> K[用户阅读并reaction反馈]:::todo
    K --> L[检测reaction并记录数据库]:::todo
    
    %% 数据聚合流程
    L --> M[定期聚合emoji到二元标签 - 喜欢/不喜欢]:::todo
    M --> N[更新preference csv文件]:::todo
    N --> O(结束):::finish
    
    %% 数据管道
    G --> P[切分parquet文件 判断reuse]:::todo
    P --> Q[上传对象存储  arxiv_id.jsonl]:::todo
    Q --> R(定期更新HuggingFace Dataset):::finish
    Q --> F
    
    %% 设置流程
    E --> S[Telegram Inline Keyboard<br/>设置Repo ID/PAT]:::todo
    S --> A
    
    %% 独立summarize流程
    T([/summarize 指令手动触发])
    T --> U[后端 src/bot/tg.py summarize]
    U --> C
    
    classDef todo fill:#f0e68c,stroke:#daa520,stroke-dasharray:5 5
    classDef finish fill:#98fb98,stroke:#32cd32
    classDef warning fill:#ffb6c1,stroke:#dc143c
    classDef action fill:#e6e6fa,stroke:#9370db
    
    class O finish
    class H,E warning
    class F action
```