# 【产品线级】CANN开放仓C++编程规范（建议稿）

## 说明

本规范以[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)为基础，参考MindSpore社区、华为通用编码规范、安全编程规范，并结合业界共识整理而成，参与CANN开源社区项目的开发者首先需要遵循本规范内容，其余遵循Google C++ Style Guide规范；
如果对规则异议，建议提交issue并说明理由，经CANN运营团队评审后可接纳并修改生效；

## 适用范围

CANN相关开源仓

---

### 1. 代码风格

#### 1.1 命名

__驼峰风格（CamelCase）__
大小写字母混用，单词连在一起，不同单词间通过单词首字母大写来分开。
按连接后的首字母是否大写，又分：大驼峰（UpperCamelCase）和小驼峰（lowerCamelCase）

| 类型 | 命名风格 |
| --- | --- |
| 类类型，结构体类型，枚举类型，联合体类型等类型定义，作用域名称 | 大驼峰 |
| 函数（包括全局函数，作用域函数，成员函数） | 大驼峰 |
| 全局变量（包括全局和命名空间域下的变量，类静态变量），局部变量，函数参数，类、结构体和联合体中的成员变量 | 小驼峰 |
| 宏，常量（const），枚举值，goto标签 | 全大写，下划线分割 |

注意：
上表中__常量__是指全局作用域、namespace域、类的静态成员域下，以const或constexpr修饰的基本数据类型、枚举、字符串类型的变量，不包括数组和其他类型变量。
上表中__变量__是指除常量定义以外的其他变量，均使用小驼峰风格。

##### 规则 1.1.1 C++文件使用小写+下划线的方式命名，以.cpp结尾，头文件以.h结尾

目前业界还有一些其他的后缀的表示方法：
- 头文件：.hh, .hpp, .hxx
- cpp文件：.cc, .cxx, .c

如果当前项目组使用了某种特定的后缀，那么可以继续使用，但是请保持风格统一。
但是对于本文档，我们默认使用.h和.cpp作为后缀。

##### 规则 1.1.2 函数命名统一使用大驼峰风格，一般采用动词或者动宾结构。

```cpp
class List {
public:
	void AddElement(const Element& element);
	Element GetElement(const unsigned int index) const;
	bool IsEmpty() const;
};

namespace Utils {
    void DeleteUser();
}
```

##### 规则 1.1.3 类型命名采用大驼峰命名风格。

所有类型命名——类、结构体、联合体、类型定义（typedef）、枚举——使用相同约定，例如：

```cpp
// classes, structs and unions
class UrlTable { ...
struct UrlTableProperties { ...
union Packet { ...
// typedefs
typedef std::map<std::string, UrlTableProperties*> PropertiesMap;
// enums
enum UrlTableErrors { ...
```

对于命名空间的命名，建议使用大驼峰：

```cpp
// namespace
namespace FileUtils {
}
```

##### 规则 1.1.4 通用变量命名采用小驼峰，包括全局变量，函数形参，局部变量，成员变量。

```cpp
std::string tableName;  // Good: 推荐此风格
std::string tablename;  // Bad: 禁止此风格
std::string path;       // Good: 只有一个单词时，小驼峰为全小写
```

全局变量应增加'g_'前缀，静态变量命名不需要加特殊前缀
全局变量是应当尽量少使用的，使用时应特别注意，所以加上前缀用于视觉上的突出，促使开发人员对这些变量的使用更加小心。
- 全局静态变量命名与全局变量相同。
- 函数内的静态变量命名与普通局部变量相同。
- 类的静态成员变量和普通成员变量相同。

```cpp
int g_activeConnectCount;

void Func()
{
    static int packetCount = 0;
    ...
}
```

类的成员变量命名以小驼峰加后下划线组成

```cpp
class Foo {
private:
    std::string fileName_;   // 添加_后缀，类似于K&R命名风格
};
```

##### 规则 1.1.5 宏、枚举值采用全大写，下划线连接的格式。

全局作用域内，有名和匿名namespace内的const常量，类的静态成员常量，全大写，下划线连接；函数局部const常量和类的普通const成员变量，使用小驼峰命名风格。

```cpp
#define MAX(a, b)   (((a) < (b)) ? (b) : (a)) // 仅对宏命名举例，并不推荐用宏实现此类功能

enum TintColor {    // 注意，枚举类型名用大驼峰，其下面的取值是全大写，下划线相连
    RED,
    DARK_RED,
    GREEN,
    LIGHT_GREEN
};

int Func(...)
{
    const unsigned int bufferSize = 100;    // 函数局部常量
    char *p = new char[bufferSize];
    ...
}

namespace Utils {
	const unsigned int DEFAULT_FILE_SIZE_KB = 200;        // 全局常量
}

```

#### 1.2 格式

##### 建议 1.2.1 行宽不超过120个字符

建议每行字符数不要超过120个。如果超过120个字符，请选择合理的方式进行换行。

例外：
- 如果一行注释包含了超过120个字符的命令或URL，则可以保持一行，以方便复制、粘贴和通过grep查找；
- 包含长路径的#include语句可以超出120个字符，但是也需要尽量避免；
- 编译预处理中的error信息可以超出一行。
预处理的error信息在一行便于阅读和理解，即使超过120个字符。

```cpp
#ifndef XXX_YYY_ZZZ
#error Header aaaa/bbbb/cccc/abc.h must only be included after xxxx/yyyy/zzzz/xyz.h, because xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#endif
```

##### 规则 1.2.2 使用空格进行缩进，每次缩进4个空格

只允许使用空格（space）进行缩进，每次缩进为4个空格。不允许使用Tab符进行缩进。
当前几乎所有的集成开发环境（IDE）都支持配置将Tab符自动扩展为4空格输入；请配置你的IDE支持使用空格进行缩进。

##### 规则 1.2.3 在声明指针、引用变量或参数时，`&`、`*`跟随变量名，另外一边留空格

```cpp
char *c;
  const std::string &str;
```

##### 规则 1.2.4 if语句必须要使用大括号

我们要求if语句都需要使用大括号，即便只有一条语句。
理由：
- 代码逻辑直观，易读；
- 在已有条件语句代码上增加新代码时不容易出错；
- 对于在if语句中使用函数式宏时，有大括号保护不易出错（如果宏定义时遗漏了大括号）。

```cpp
// 即使if分支代码只有一行，也必须使用大括号
if (cond) {
  single line code;
}
```

##### 规则 1.2.5 for/while等循环语句必须使用大括号

和条件表达式类似，我们要求for/while循环语句必须加上大括号，即便循环体是空的，或循环语句只有一条。

```cpp
for (int i = 0; i < someRange; i++) {   // Good: 使用了大括号
    DoSomething();
}
```

```cpp
while (condition) { }   // Good：循环体是空，使用大括号
```

##### 规则 1.2.6 表达式换行要保持换行的一致性，运算符放行末

较长的表达式，不满足行宽要求的时候，需要在适当的地方换行。一般在较低优先级运算符或连接符后面截断，运算符或连接符放在行末。
运算符、连接符放在行末，表示"未结束，后续还有"。
例：

```cpp
// 假设下面第一行已经不满足行宽要求
if ((currentValue > threshold) &&  // Good：换行后，逻辑操作符放在行尾
    someCondition) {
    DoSomething();
    ...
}

int result = reallyReallyLongVariableName1 +    // Good
             reallyReallyLongVariableName2;
```

表达式换行后，注意保持合理对齐，或者4空格缩进。参考下面例子

```cpp
int sum = longVariableName1 + longVariableName2 + longVariableName3 +
    longVariableName4 + longVariableName5 + longVariableName6;         // Good: 4空格缩进

int sum = longVariableName1 + longVariableName2 + longVariableName3 +
          longVariableName4 + longVariableName5 + longVariableName6;   // Good: 保持对齐
```

##### 规则 1.2.7 使用K&R缩进风格

__K&R风格__
换行时，函数（不包括lambda表达式）左大括号另起一行放行首，并独占一行；其他左大括号跟随语句放行末。
右大括号独占一行，除非后面跟着同一语句的剩余部分，如do语句中的while，或者if语句的else/else if，或者逗号、分号。

如：

```cpp
struct MyType {     // 跟随语句放行末，前置1空格
    ...
};

int Foo(int a)
{                   // 函数左大括号独占一行，放行首
    if (...) {
        ...
    } else {
        ...
    }
}
```

推荐这种风格的理由：

- 代码更紧凑；
- 相比另起一行，放行末使代码阅读节奏感上更连续；
- 符合后来语言的习惯，符合业界主流习惯；
- 现代集成开发环境（IDE）都具有代码缩进对齐显示的辅助功能，大括号放在行尾并不会对缩进和范围产生理解上的影响。

对于空函数体，可以将大括号放在同一行：

```cpp
class MyClass {
public:
    MyClass() : value_(0) {}

private:
    int value_;
};
```

##### 规则 1.2.8 多个变量定义和赋值语句不允许写在一行

每行只有一个变量初始化的语句，更容易阅读和理解。

##### 规则 1.2.9 合理安排空行，保持代码紧凑

减少不必要的空行，可以显示更多的代码，方便代码阅读。下面有一些建议遵守的规则：
- 根据上下内容的相关程度，合理安排空行；
- 函数内部、类型定义内部、宏内部、初始化表达式内部，不使用连续空行
- 不使用连续**3**个空行，或更多
- 大括号内的代码块行首之前和行尾之后不要加空行，但namespace的大括号内不作要求。

```cpp
int Foo()
{
    ...
}



int Bar()  // Bad：最多使用连续2个空行。
{
    ...
}


if (...) {
        // Bad：大括号内的代码块行首不要加入空行
    ...
        // Bad：大括号内的代码块行尾不要加入空行
}

int Foo(...)
{
        // Bad：函数体内行首不要加空行
    ...
}
```

#### 1.3 注释

一般的，尽量通过清晰的架构逻辑，好的符号命名来提高代码可读性；需要的时候，才辅以注释说明。
注释是为了帮助阅读者快速读懂代码，所以要从读者的角度出发，**按需注释**。

注释内容要简洁、明了、无二义性，信息全面且不冗余。

在C++代码中，使用 `/*` `*/` 和 `//` 都是可以的。
按注释的目的和位置，注释可分为不同的类型，如文件头注释、函数头注释、代码注释等等；
同一类型的注释应该保持统一的风格。

##### 规则 1.3.1 文件头注释包含版权声明

如下例子：

```cpp
/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

```

> 关于版权说明，应注意：
> 2025年新建的文件，应该是 `Copyright (c) 2025 Huawei Technologies Co., Ltd.`

##### 规则 1.3.2 代码注释置于对应代码的上方或右边，注释符与注释内容之间要有1个空格，右置注释与前面代码至少1空格，使用 `//`，而不是 `/**/`

```cpp
// this is multi-
// line comment
int foo; // this single-line comment
```

##### 规则 1.3.3 代码中禁止使用TODO/TBD/FIXME等注释，建议提issue跟踪

##### 建议 1.3.4 不要写空有格式的函数头注释

并不是所有的函数都需要函数头注释，函数尽量通过函数名自注释，按需写函数头注释；函数原型无法表达的，却又希望读者知道的信息，才需要加函数头注释辅助说明。
不要写无用、信息冗余的函数头，函数头注释内容可选，但不限于：功能说明、返回值，性能约束、用法、内存约定、算法实现、可重入的要求等。
例：

```cpp
/*
 * 返回实际写入的字节数，-1表示写入失败
 * 注意，内存 buf 由调用者负责释放
 */
int WriteString(const char *buf, int len);
```

坏的例子：

```cpp
/*
 * 函数名：WriteString
 * 功能：写入字符串
 * 参数：
 * 返回值：
 */
int WriteString(const char *buf, int len);
```

上面例子中的问题：

- 参数、返回值，空有格式没内容
- 函数名信息冗余
- 关键的buf由谁释放没有说清楚

##### 建议 1.3.5 不用的代码段直接删除，不要注释掉

被注释掉的代码，无法被正常维护；当企图恢复使用这段代码时，极有可能引入易被忽略的缺陷。
正确的做法是，不需要的代码直接删除掉。若再需要时，考虑移植或重写这段代码。

这里说的注释掉代码，包括用 /* */ 和 //，还包括 #if 0，#ifdef NEVER_DEFINED等等。

### 2. 通用编码

#### 2.1 代码设计

##### 规则 2.1.1 对所有外部数据进行合法性检查，包括但不限于：函数入参、外部输入命名行、文件、环境变量、用户数据等

##### 规则 2.1.2 函数执行结果传递，优先使用返回值，尽量避免使用出参

```cpp
FooBar *Func(const std::string &in);
```

##### 规则 2.1.3 删除无效、冗余或永不执行的代码

虽然大多数现代编译器在许多情况下可以对无效或从不执行的代码告警，响应告警应识别并清除告警；
应该主动识别无效的语句或表达式，并将其从代码中删除。

##### 规则 2.1.4 补充C++异常机制的规范

###### 规则 2.1.4.1 需要指定捕获异常种类，禁止捕获所有异常

```cpp
// 错误示范
try {
  // do something;
} catch (...) {
  // do something;
}
// 正确示范
try {
  // do something;
} catch (const std::bad_alloc &e) {
  // do something;
}
```

#### 2.2 头文件和预处理

##### 规则 2.2.1 使用新的标准C++头文件

```cpp
// 正确示范
#include <cstdlib>
// 错误示范
#include <stdlib.h>
```

##### 规则 2.2.2 禁止头文件循环依赖

头文件循环依赖，指a.h包含b.h，b.h包含c.h，c.h包含a.h之类导致任何一个头文件修改，都导致所有包含了a.h/b.h/c.h的代码全部重新编译一遍。
头文件循环依赖直接体现了架构设计上的不合理，可通过优化架构去避免。

##### 规则 2.2.3 禁止包含用不到的头文件

##### 规则 2.2.4 禁止通过extern声明的方式引用外部函数接口、变量

##### 规则 2.2.5 禁止在extern "C"中包含头文件

##### 规则 2.2.6 禁止在头文件中或者#include之前使用using导入命名空间

#### 2.3 数据类型

##### 建议 2.3.1 避免滥用typedef或者#define对基本类型起别名

##### 规则 2.3.2 使用using而非typedef定义类型的别名，避免类型变化带来的散弹式修改

```cpp
// 正确示范
using FooBarPtr = std::shared_ptr<FooBar>;
// 错误示范
typedef std::shared_ptr<FooBar> FooBarPtr;
```

#### 2.4 常量

##### 规则 2.4.1 禁止使用宏表示常量

##### 规则 2.4.2 禁止使用魔鬼数字\字符串

##### 建议 2.4.3 建议每个常量保证单一职责

#### 2.5 变量

##### 规则 2.5.1 优先使用命名空间来管理全局常量，如果和某个class有直接关系的，可以使用静态成员常量

```cpp
namespace foo {
  int kGlobalVar;

  class Bar {
    private:
      static int static_member_var_;
  };
}
```

##### 规则 2.5.2 尽量避免使用全局变量，谨慎使用单例模式，避免滥用

##### 规则 2.5.3 禁止在变量自增或自减运算的表达式中再次引用该变量

##### 规则 2.5.4 指向资源句柄或描述符的指针变量在资源释放后立即赋予新值或置为NULL

##### 规则 2.5.5 禁止使用未经初始化的变量

#### 2.6 表达式

##### 建议 2.6.1 表达式的比较遵循左侧倾向于变化、右侧倾向于不变的原则

```cpp
// 正确示范
if (ret != SUCCESS) {
  ...
}

// 错误示范
if (SUCCESS != ret) {
  ...
}
```

##### 规则 2.6.2 通过使用括号明确操作符的优先级，避免出现低级错误

```cpp
// 正确示范
if (cond1 || (cond2 && cond3)) {
  ...
}

// 错误示范
if (cond1 || cond2 && cond3) {
  ...
}
```

#### 2.7 转换

##### 规则 2.7.1 使用有C++提供的类型转换，而不是C风格的类型转换，避免使用const_cast和reinterpret_cast

#### 2.8 控制语句

##### 规则 2.8.1 switch语句要有default分支

#### 2.9 声明与初始化

##### 规则 2.9.1 禁止用`memcpy_s`、`memset_s`初始化非POD对象

#### 2.10 指针和数组

##### 规则 2.10.1 禁止持有std::string的c_str()返回的指针

```cpp
// 错误示范
const char * a = std::to_string(12345).c_str();
```

##### 规则 2.10.2 优先使用unique_ptr而不是shared_ptr

##### 规则 2.10.3 使用std::make_shared而不是new创建shared_ptr

```cpp
// 正确示范
std::shared_ptr<FooBar> foo = std::make_shared<FooBar>();
// 错误示范
std::shared_ptr<FooBar> foo(new FooBar());
```

##### 规则 2.10.4 使用智能指针管理对象，避免使用new/delete

##### 规则 2.10.5 禁止使用auto_ptr

##### 规则 2.10.6 对于指针和引用类型的形参，如果是不需要修改的，要求使用const

##### 规则 2.10.7 数组作为函数参数时，必须同时将其长度作为函数的参数

```cpp
int ParseMsg(BYTE *msg, size_t msgLen) {
  ...
}
```

#### 2.11 字符串

##### 规则 2.11.1 对字符串进行存储操作，确保字符串有'\0'结束符

#### 2.12 断言

##### 规则 2.12.1 断言不能用于校验程序在运行期间可能导致的错误，可能发生的运行错误要用错误处理代码来处理

#### 2.13 类和对象

##### 规则 2.13.1 单个对象释放使用delete，数组对象释放使用delete[]

```cpp
const int kSize = 5;
int *number_array = new int[kSize];
int *number = new int();
...
delete[] number_array;
number_array = nullptr;
delete number;
number = nullptr;
```

##### 规则 2.13.2 禁止使用std::move操作const对象

##### 规则 2.13.3 严格使用virtual/override/final修饰虚函数

```cpp
class Base {
  public:
    virtual void Func();
};

class Derived : public Base {
  public:
    void Func() override;
};

class FinalDerived : public Derived {
  public:
    void Func() final;
};
```

#### 2.14 函数设计

##### 规则 2.14.1 使用RAII特性来帮助追踪动态分配

```cpp
// 正确示范
{
  std::lock_guard<std::mutex> lock(mutex_);
  ...
}
```

##### 规则 2.14.2 非局部范围使用lambdas时，避免按引用捕获

```cpp
{
  int local_var = 1;
  auto func = [&]() { ...; std::cout << local_var << std::endl; };
  thread_pool.commit(func);
}
```

##### 规则 2.14.3 禁止虚函数使用缺省参数值

##### 建议 2.14.4 使用强类型参数\成员变量，避免使用void*

#### 2.15 函数使用

##### 规则 2.15.1 函数传参传递，要求入参在前，出参在后

```cpp
bool Func(const std::string &in, FooBar *out1, FooBar *out2);
```

##### 规则 2.15.2 函数传参传递，要求入参用`const T &`，出参用`T *`

```cpp
bool Func(const std::string &in, FooBar *out1, FooBar *out2);
```

##### 规则 2.15.3 函数传参传递，不涉及所有权的场景，使用T *或const T &作为参数，而不是智能指针

```cpp
// 正确示范
  bool Func(const FooBar &in);
  // 错误示范
  bool Func(std::shared_ptr<FooBar> in);
```

##### 规则 2.15.4 函数传参传递，如需传递所有权，建议使用shared_ptr+move传参

```cpp
class Foo {
  public:
    explicit Foo(shared_ptr<T> x):x_(std::move(x)){}
  private:
    shared_ptr<T> x_;
};
```

##### 规则 2.15.5 单参数构造函数必须用explicit修饰，多参数构造函数禁止使用explicit修饰

```cpp
explicit Foo(int x);          //good :white_check_mark:
  explicit Foo(int x, int y=0); //good :white_check_mark:
  Foo(int x, int y=0);          //bad  :x:
  explicit Foo(int x, int y);   //bad  :x:
```

##### 规则 2.15.6 拷贝构造和拷贝赋值操作符应该是成对出现或者禁止

```cpp
class Foo {
  private:
    Foo(const Foo&) = default;
    Foo& operator=(const Foo&) = default;
    Foo(Foo&&) = delete;
    Foo& operator=(Foo&&) = delete;
};
```

##### 规则 2.15.7 禁止保存、delete指针参数

##### 规则 2.15.8 禁止使用非安全函数，需要给出清单

##### 规则 2.15.9 禁止使用非安全退出函数，需要给出清单

```cpp
{
  kill(...);            // 调用kill强行终止其他进程(如kill -9)，会导致其他进程的资源得不到清理。
  TerminateProcess();   // 调用TerminateProcess函数强行终止其他进程，会导致其他进程的资源得不到清理。
  pthread_exit();       // 严禁在线程内主动终止自身线程，线程函数在执行完毕后会自动、安全地退出;
  ExitThread();         // 严禁在线程内主动终止自身线程，线程函数在执行完毕后会自动、安全地退出;
  exit();               // main函数以外，禁止任何地方调用，程序应该安全退出；
  ExitProcess();        // main函数以外，禁止任何地方调用，程序应该安全退出；
  abort();              // 禁用，abort会导致程序立即退出，资源得不到清；
}
```

##### 规则 2.15.10 禁用rand函数产生用于安全用途的伪随机数

C标准库rand()函数生成的是伪随机数，请使用/dev/random生成随机数。

##### 规则 2.15.11 严禁使用string类存储敏感信息

string类是C++内部定义的字符串管理类，如果口令等敏感信息通过string进行操作，在程序运行过程中，敏感信息可
能会散落到内存的各个地方，并且无法清0。

以下代码，Foo函数中获取密码，保存到string变量password中，随后传递给VerifyPassword函数，在这个过程中，
password实际上在内存中出现了2份。

```cpp
int VerifyPassword(string password) {
  //...
}
int Foo() {
  string password = GetPassword();
  VerifyPassword(password);
  ...
}
```

应该使用char或unsigned char保存敏感信息，如下代码：

```cpp
int VerifyPassword(const char *password) {
  //...
}
int Foo() {
  char password[MAX_PASSWORD] = {0};
  GetPassword(password, sizeof(password));
  VerifyPassword(password);
  ...
}
```

##### 规则 2.15.12 内存中的敏感信息使用完毕后立即清0

口令、密钥等敏感信息使用完毕后立即清0，避免被攻击者获取。

#### 2.16 内存

##### 规则 2.16.1 内存分配后必须判断是否成功

内存分配失败后，那么后续的操作存在未定义的行为风险。比如malloc申请失败返回了空指针，对空指针的解引用是一种未定义行为。

##### 规则 2.16.2 禁止引用未初始化的内存

malloc、new分配出来的内存没有被初始化为0，要确保内存被引用前是被初始化的。

##### 规则 2.16.3 避免使用realloc()函数

随着参数的不同，realloc函数行为也不同，这不是一个设计良好的函数。虽然在编码中提供了一些便利性，但是却极易引发各种bug。

##### 规则 2.16.4 不要使用alloca()函数申请栈上内存

POSIX和C99均未定义alloca()的行为，在有些平台下不支持该函数，使用alloca会降低程序的兼容性和可移植性，该函数在栈帧里申请内存，申请的大小很可能超过栈的边界，影响后续的代码执行。

#### 2.17 文件

##### 规则 2.17.1 必须对文件路径进行规范化后再使用

当文件路径来自外部数据时，需要先将文件路径规范化，如果没有作规范化处理，攻击者就有机会通过恶意构造文件路径进行文件的越权访问：
例如，攻击者可以构造"../../../etc/passwd"的方式进行任意文件访问。
在Linux下，使用realpath函数，在Windows下，使用PathCanonicalize函数进行文件路径的规范化。

【错误代码示例】
以下代码从外部获取到文件名称，拼接成文件路径后，直接对文件内容进行读取，导致攻击者可以读取到任意文件的内容：

```cpp
char *fileName = GetMsgFromRemote();
...
sprintf_s(untrustPath, sizeof(untrustPath), "/tmp/%s", fileName);
char *text = ReadFileContent(untrustPath);   // Bad，读取前未检查untrustPath是否允许访问
```

【正确代码示例】
正确的做法是，对路径进行规范化后，再判断路径是否是本程序所认为的合法的路径：

```cpp
char *fileName = GetMsgFromRemote();
...
sprintf_s(untrustPath, sizeof(untrustPath), "/tmp/%s", fileName);
char path[PATH_MAX] = {0};
if (realpath(untrustPath, path) == NULL) {
    //error
    ...
}
if (!IsValidPath(path)) {    // Good，检查文件位置是否正确
    //error
    ...
}
char *text = ReadFileContent(untrustPath);
```

【例外】
运行于控制台的命令行程序，通过控制台手工输入文件路径，可以作为本建议例外。

##### 规则 2.17.2 不要在共享目录中创建临时文件

程序的临时文件应当是程序自身独享的，任何将自身临时文件置于共享目录的做法，将导致其他共享用户获得该程序的额外信息，产生信息泄露。因此，不要在任何共享目录创建仅由程序自身使用的临时文件。
如Linux下的/tmp目录是一个所有用户都可以访问的共享目录，不应在该目录下创建仅由程序自身使用的临时文件。

#### 2.18 安全函数

| 安全函数类型 | 说明 | 备注 |
| --- | --- | --- |
| xxx_s | Huawei Secure C库的安全函数API | 集成Huawei Secure C库即可使用 |
| xxx_sp | Huawei Secure C库的安全函数性能优化API（宏实现） | 性能优化宏接口对count、destMax、strSrc为常量时有优化效果，如果是变量则优化效果不明显。宏接口使用策略：默认使用\_s接口，在性能敏感的调用点受限使用\_sp接口，受限场景如下：a) memset\_sp/memcpy\_sp使用场景：destMax和count为常量；b) strcpy\_sp/strcat\_sp使用场景：destMax为常量且strSrc为字面量；c) strncpy\_sp/strncat\_sp使用场景：destMax和count为常量且strSrc为字面量 |

##### 规则 2.18.1 请使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数

| 函数类别 | 危险函数 | 安全替代函数 |
| --- | --- | --- |
| 内存拷贝 | memcpy或bcopy | memcpy_s |
|  | wmemcpy | wmemcpy_s |
|  | memmove | memmove_s |
|  | wmemmove | wmemmove_s |
| 字符串拷贝 | strcpy | strcpy_s |
|  | wcscpy | wcscpy_s |
|  | strncpy | strncpy_s |
|  | wcsncpy | wcsncpy_s |
| 字符串串接 | strcat | strcat_s |
|  | wcscat | wcscat_s |
|  | strncat | strncat_s |
|  | wcsncat | wcsncat_s |
| 格式化输出 | sprintf | sprintf_s |
|  | swprintf | swprintf_s |
|  | vsprintf | vsprintf_s |
|  | vswprintf | vswprintf_s |
|  | snprintf | snprintf_s 或 snprintf_truncated_s |
|  | vsnprintf | vsnprintf_s 或 vsnprintf_truncated_s |
| 格式化输入 | scanf | scanf_s |
|  | wscanf | wscanf_s |
|  | vscanf | vscanf_s |
|  | vwscanf | vwscanf_s |
|  | fscanf | fscanf_s |
|  | fwscanf | fwscanf_s |
|  | vfscanf | vfscanf_s |
|  | vfwscanf | vfwscanf_s |
|  | sscanf | sscanf_s |
|  | swscanf | swscanf_s |
|  | vsscanf | vsscanf_s |
|  | vswscanf | vswscanf_s |
| 标准输入流输入 | gets | gets_s |
| 内存初始化 | memset | memset_s |

##### 规则 2.18.2 正确设置安全函数中的destMax参数

##### 规则 2.18.3 禁止封装安全函数

##### 规则 2.18.4 禁止用宏重命名安全函数

```cpp
#define XXX_memcpy_s memcpy_s
#define SEC_MEM_CPY memcpy_s
#define XX_memset_s(dst, dstMax, val, n) memset_s((dst), (dstMax), (val), (n))
```

##### 规则 2.18.5 禁止自定义安全函数

使用宏重命名安全函数不利于静态代码扫描工具（非编译型）定制针对安全函数误用的规则，同时，由于命名风格多
样，也不利于提示代码开发者函数的真实用途，容易造成对代码的误解及重命名安全函数的误用。重命名安全函数不
会改变安全函数本身的检查能力。

```cpp
void MemcpySafe(void *dest, unsigned int destMax, const void *src, unsigned int count) {
  ...
}
```

##### 规则 2.18.6 必须检查安全函数返回值，并进行正确的处理

原则上，如果使用了安全函数，需要进行返回值检查。如果返回值!=EOK，那么本函数一般情况下应该立即返回，不
能继续执行。
安全函数有多个错误返回值，如果安全函数返回失败，在本函数返回前，根据产品具体场景，可以做如下操作（执行
其中一个或多个措施）：
（1）记录日志
（2）返回错误
（3）调用abort立即退出程序

```cpp
{
  ...
  err = memcpy_s(destBuff, destMax, src, srcLen);
  if (err != EOK) {
    MS_LOG("memcpy_s failed, err = %d\n", err);
    return FALSE;
  }
  ...
}
```

##### 规则 2.18.7 禁止外部可控数据作为system、popen、WinExec、ShellExecute、execl、xeclp、execle、execv、execvp、CreateProcess等进程启动函数的参数

##### 规则 2.18.8 禁止外部可控数据作为dlopen/LoadLibrary等模块加载函数的参数

##### 规则 2.18.9 禁止在信号处理例程中调用非异步安全函数

信号处理例程应尽可能简化。在信号处理例程中如果调用非异步安全函数，可能会导致函数的执行不符合预期的结
果。下列代码中的信号处理程序通过调用fprintf()写日志，但该函数不是异步安全函数。

```cpp
void Handler(int sigNum) {
  ...
  fprintf(stderr, "%s\n", info);
}
```

### 3. 安全编码

#### 3.1 总体原则

##### 规则 3.1.1 保证静态类型安全

C++应该是静态类型安全的，这样可以减少运行时的错误，提升代码的健壮性。但是由于C++存在下面的特性，会破坏C++静态类型安全，针对这部分特性要仔细处理：

- 联合体
- 类型转换
- 缩窄转换
- 类型退化
- 范围错误
- void* 类型指针

可以通过约束这些特性的使用，或者使用C++的新特性，例如std::variant（C++17）、std::span（C++20）等来解决这些问题，提升C++代码的健壮性。

##### 规则 3.1.2 保证内存安全

C++语言的内存完全由程序员自己控制，所以在操作内存的时候必须保证内存安全，防止出现内存错误：

- 内存越界访问
- 释放以后继续访问内存
- 解引用空指针
- 内存没有初始化
- 把指向局部变量的引用或者指针传递到了函数外部或者或者其他线程中
- 申请的内存或者资源没有及时释放

建议使用更加安全的C++的特性，比如RAII、引用、智能指针等，来提升代码的健壮性。

##### 规则 3.1.3 禁止使用编译器"未定义行为"

遵循ISO C++标准，标准中未定义的行为禁止使用。对于编译器实现的特性或者GCC等编译器提供的扩展特性也需要谨慎使用，这些特性会降低代码的可移植性。

#### 3.2 类

##### 规则 3.2.1 delete操作符、移动构造函数、移动赋值操作符、swap函数应该有noexcept声明

#### 3.3 表达式与语句

##### 规则 3.3.1 禁止逐位操作非trivially copyable对象

#### 3.4 资源管理

##### 规则 3.4.1 new和delete配对使用，new[]和delete[]配对使用

##### 规则 3.4.2 自定义new/delete操作符需要配对定义，且行为与被替换的操作符一致

##### 规则 3.4.3 使用恰当的方式处理new操作符的内存分配错误

##### 规则 3.4.4 避免出现delete this操作

#### 3.5 标准库

##### 规则 3.5.1 禁止从空指针创建std::string

##### 规则 3.5.2 不要保存std::string类型的 `c_str` 和 `data` 成员函数返回的指针

##### 规则 3.5.3 避免使用atoi、atol、atoll、atof函数

##### 规则 3.5.4 禁止使用std::string存储敏感信息

##### 规则 3.5.5 调用格式化输入/输出函数时，禁止format参数受外部数据控制

##### 规则 3.5.6 禁用程序与线程的退出函数和atexit函数

##### 规则 3.5.7 禁止调用kill、TerminateProcess函数直接终止其他进程
