# Python 核心审查规则

## 必检规则

对diff中每个函数、每个资源操作、每个外部数据处理逐条检查。命中即为【严重】，不可降级：

1. 类型混淆：`None`返回值是否被当作有效对象使用？`Optional`类型是否先检查再访问？
2. 异常吞没：`except Exception`/`except:`是否隐藏了关键错误？异常处理是否过于宽泛？
3. 资源泄漏：文件/连接/锁是否使用`with`语句或`try/finally`确保释放？
4. 可变默认参数：函数默认参数是否使用了`list`/`dict`/`set`等可变对象？
5. 注入风险：是否用`eval()`/`exec()`/`os.system()`处理外部输入？SQL是否用参数化查询？
6. 路径遍历：外部输入的文件路径是否做了规范化和白名单校验？
7. 并发安全：多线程共享数据是否有竞争条件？GIL不保护跨语句的原子性
8. 除零保护：除数是否可能为零？
9. 索引越界：列表/元组索引是否有边界保护？字典key是否可能不存在？

## 分层规则

**【严重】** — 必检规则命中 + 敏感信息硬编码（密码/密钥/token写在代码中）+ pickle/yaml.load反序列化不可信数据 + subprocess.shell=True拼接外部输入

**【一般】** — 裸`except:`未re-raise + 未使用`logging`而用`print`做日志 + 全局可变状态 + 函数超过50行 + 嵌套超过4层 + 未使用类型注解（公共API）+ `import *` + 循环导入

**【建议】** — PEP 8风格违规 + 魔鬼数字/字符串 + 冗余代码 + TODO/FIXME + 缺少docstring（公共API）

## 命名规范速查（PEP 8）

| 类型 | 风格 | 示例 |
|------|------|------|
| 模块/包 | 小写下划线 | `my_module`, `utils` |
| 类 | 大驼峰 | `MyClass`, `HTTPServer` |
| 函数/方法 | 小写下划线 | `get_value`, `calculate_sum` |
| 变量 | 小写下划线 | `total_count`, `file_path` |
| 常量 | 全大写下划线 | `MAX_RETRY`, `DEFAULT_TIMEOUT` |
| 私有属性 | 前置下划线 | `_internal_state` |
| 名称修饰 | 双前置下划线 | `__private_method` |

## 常见反模式速查

| 反模式 | 修正 |
|--------|------|
| `except:`/`except Exception:` | 指定具体异常类型 |
| `def f(x=[]):` | `def f(x=None): x = x or []` |
| `eval(user_input)` | 使用`ast.literal_eval`或白名单解析 |
| `os.system(cmd)` | `subprocess.run([...], shell=False)` |
| `open(f)`无close | `with open(f) as fh:` |
| `== None` | `is None` |
| `type(x) == int` | `isinstance(x, int)` |
