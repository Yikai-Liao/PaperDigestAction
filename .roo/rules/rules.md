* 请使用英文思考，编程，写注释，但是使用中文进行对话交流
* 我使用高效且现代化的工具，如lsd,tokei,polars,numpy,duckdb,fastapi,uv,loguru，fd,避免使用老旧的有问题的工具如pandas,conda,find,直接使用pip(使用uv add),不要使用python自带的logging
* 你可以通过git 工具来查看文件变更，历史，还原文件
* 你不能通过编辑pyproject.toml 来添加依赖，使用uv add remove 来管理依赖
* 我喜欢使用toml管理配置文件，使用pydantic创建对应的config类进行管理
* 我喜欢简洁，少嵌套的，易于维护，可复用的，面向对象的代码，并带有完善的注释，但是不能啰唆
* 不能面向测试编程，将答案硬编码在代码中
* 不能为了通过测试，而简化测试例子，绕过需要处理的复杂复杂情况
* 即使是python，也要写完整的type hint，并且你需要相信type hint，不应该写大量处理代码，来兼容非法输入，应该直接对非法输入报错
* 倾向与使用no-progress来减少命令行冗余输出，例如uv 使用--no-progress
* 任何开发都要有限考虑跨平台兼容，例如不要依赖make，而是使用其他跨平台方案
* 尽量减少一个项目使用的技术栈与依赖，如果使用一个包解决的问题不要引入新的依赖
* 使用python时，多使用pathlib.Path，多通过__file__，利用文件相对管理来添加path解决导入失败的问题
* 在python中，使用pytest来进行单元测试，并且要严格注意文件import路径问题
* 你可以在终端通过claude -p "你的自然语言指令" 来调用claude 来帮你完成一些简单的小任务，但是注意不要让他做复杂任务
