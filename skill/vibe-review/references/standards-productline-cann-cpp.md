# 【产品线级】CANN开放仓C++编程规范（建议稿）

## 说明

本规范以[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)为基础，参考MindSpore社区、华为通用编码规范、安全编程规范，并结合业界共识整理而成，参与CANN开源社区项目的开发者首先需要遵循本规范内容，其余遵循Google C++ Style Guide规范；
如果对规则异议，建议提交issue并说明理由，经CANN运营团队评审后可接纳并修改生效；

## 适用范围

CANN相关开源仓

---

### 1. 代码风格

#### 1.1命名

__驼峰风格(CamelCase)__
大小写字母混用，单词连在一起，不同单词间通过单词首字母大写来分开。
按连接后的首字母是否大写，又分：大驼峰(UpperCamelCase)和小驼峰(lowerCamelCase)

| 类型                                                                                                     | 命名风格           |
| -------------------------------------------------------------------------------------------------------- | ------------------ |
| 类类型，结构体类型，枚举类型，联合体类型等类型定义，作用域名称                                           | 大驼峰             |
| 函数（包括全局函数，作用域函数，成员函数）                                                               | 大驼峰             |
| 全局变量（包括全局和命名空间域下的变量，类静态变量），局部变量，函数参数，类、结构体和联合体中的成员变量 | 小驼峰             |
| 宏，常量(const)，枚举值，goto标签                                                                        | 全大写，下划线分割 |

注意：
上表中__常量__是指全局作用域、namespace域、类的静态成员域下，以const或constexpr修饰的基本数据类型、枚举、字符串类型的变量，不包括数组和其他类型变量。
上表中__变量__是指除常量定义以外的其他变量，均使用小驼峰风格。

##### 规则1.1.1 C++文件使用小写+下划线的方式命名，以.cpp结尾，头文件以。h结尾

目前业界还有一些其他的后缀的表示方法：
- 头文件：.hh, .hpp, .hxx
- cpp文件：.cc, .cxx, .c

如果当前项目组使用了某种特定的后缀，那么可以继续使用，但是请保持风格统一。
但是对于本文档，我们默认使用。h和.cpp作为后缀。

##### 规则1.1.2函数命名统一使用大驼峰风格，一般采用动词或者动宾结构。

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

##### 规则1.1.3类型命名采用大驼峰命名风格。

所有类型命名——类、结构体、联合体、类型定义(typedef)、枚举——使用相同约定，例如：

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

##### 规则1.1.4通用变量命名采用小驼峰，包括全局变量，函数形参，局部变量，成员变量。

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

##### 规则1.1.5宏、枚举值采用全大写，下划线连接的格式。

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

#### 1.2格式

##### 建议1.2.1行宽不超过120个字符

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

##### 规则1.2.2使用空格进行缩进，每次缩进4个空格

只允许使用空格(space)进行缩进，每次缩进为4个空格。不允许使用Tab符进行缩进。
当前几乎所有的集成开发环境(IDE)都支持配置将Tab符自动扩展为4空格输入；请配置你的IDE支持使用空格进行缩进。

##### 规则1.2.3在声明指针、引用变量或参数时，`&`、`*`跟随变量名，另外一边留空格

```cpp
char *c;
  const std::string &str;
```

##### 规则1.2.4 if语句必须要使用大括号

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

##### 规则1.2.5 for/while等循环语句必须使用大括号

和条件表达式类似，我们要求for/while循环语句必须加上大括号，即便循环体是空的，或循环语句只有一条。

```cpp
for (int i = 0; i < someRange; i++) {   // Good: 使用了大括号
    DoSomething();
}
```

```cpp
while (condition) { }   // Good：循环体是空，使用大括号
```

##### 规则1.2.6表达式换行要保持换行的一致性，运算符放行末

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

##### 规则1.2.7使用K&R缩进风格

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
- 现代集成开发环境(IDE)都具有代码缩进对齐显示的辅助功能，大括号放在行尾并不会对缩进和范围产生理解上的影响。

对于空函数体，可以将大括号放在同一行：

```cpp
class MyClass {
public:
    MyClass() : value_(0) {}

private:
    int value_;
};
```

##### 规则1.2.8多个变量定义和赋值语句不允许写在一行

每行只有一个变量初始化的语句，更容易阅读和理解。

##### 规则1.2.9合理安排空行，保持代码紧凑

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

#### 1.3注释

一般的，尽量通过清晰的架构逻辑，好的符号命名来提高代码可读性；需要的时候，才辅以注释说明。
注释是为了帮助阅读者快速读懂代码，所以要从读者的角度出发，**按需注释**。

注释内容要简洁、明了、无二义性，信息全面且不冗余。

在C++代码中，使用 `/*` `*/` 和 `//` 都是可以的。
按注释的目的和位置，注释可分为不同的类型，如文件头注释、函数头注释、代码注释等等；
同一类型的注释应该保持统一的风格。

##### 规则1.3.1文件头注释包含版权声明

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

##### 规则1.3.2代码注释置于对应代码的上方或右边，注释符与注释内容之间要有1个空格，右置注释与前面代码至少1空格，使用 `//`，而不是 `/**/`

```cpp
// this is multi-
// line comment
int foo; // this single-line comment
```

##### 规则1.3.3代码中禁止使用TODO/TBD/FIXME等注释，建议提issue跟踪

##### 建议1.3.4不要写空有格式的函数头注释

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

##### 建议1.3.5不用的代码段直接删除，不要注释掉

被注释掉的代码，无法被正常维护；当企图恢复使用这段代码时，极有可能引入易被忽略的缺陷。
正确的做法是，不需要的代码直接删除掉。若再需要时，考虑移植或重写这段代码。

这里说的注释掉代码，包括用 /* */ 和 //，还包括 #if 0，#ifdef NEVER_DEFINED等等。

### 2. 通用编码

#### 2.1代码设计

##### 规则2.1.1对所有外部数据进行合法性检查，包括但不限于：函数入参、外部输入命名行、文件、环境变量、用户数据等

##### 规则2.1.2函数执行结果传递，优先使用返回值，尽量避免使用出参

```cpp
FooBar *Func(const std::string &in);
```

##### 规则2.1.3删除无效、冗余或永不执行的代码

虽然大多数现代编译器在许多情况下可以对无效或从不执行的代码告警，响应告警应识别并清除告警；
应该主动识别无效的语句或表达式，并将其从代码中删除。

##### 规则2.1.4补充C++异常机制的规范

###### 规则2.1.4.1需要指定捕获异常种类，禁止捕获所有异常

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

#### 2.2头文件和预处理

##### 规则2.2.1使用新的标准C++头文件

```cpp
// 正确示范
#include <cstdlib>
// 错误示范
#include <stdlib.h>
```

##### 规则2.2.2禁止头文件循环依赖

头文件循环依赖，指a。h包含b.h，b.h包含c.h，c.h包含a。h之类导致任何一个头文件修改，都导致所有包含了a.h/b.h/c.h的代码全部重新编译一遍。
头文件循环依赖直接体现了架构设计上的不合理，可通过优化架构去避免。

##### 规则2.2.3禁止包含用不到的头文件

##### 规则2.2.4禁止通过extern声明的方式引用外部函数接口、变量

##### 规则2.2.5禁止在extern "C"中包含头文件

##### 规则2.2.6禁止在头文件中或者#include之前使用using导入命名空间

#### 2.3数据类型

##### 建议2.3.1避免滥用typedef或者#define对基本类型起别名

##### 规则2.3.2使用using而非typedef定义类型的别名，避免类型变化带来的散弹式修改

```cpp
// 正确示范
using FooBarPtr = std::shared_ptr<FooBar>;
// 错误示范
typedef std::shared_ptr<FooBar> FooBarPtr;
```

#### 2.4常量

##### 规则2.4.1禁止使用宏表示常量

##### 规则2.4.2禁止使用魔鬼数字\字符串

##### 建议2.4.3建议每个常量保证单一职责

#### 2.5变量

##### 规则2.5.1优先使用命名空间来管理全局常量，如果和某个class有直接关系的，可以使用静态成员常量

```cpp
namespace foo {
  int kGlobalVar;

  class Bar {
    private:
      static int static_member_var_;
  };
}
```

##### 规则2.5.2尽量避免使用全局变量，谨慎使用单例模式，避免滥用

##### 规则2.5.3禁止在变量自增或自减运算的表达式中再次引用该变量

##### 规则2.5.4指向资源句柄或描述符的指针变量在资源释放后立即赋予新值或置为NULL

##### 规则2.5.5禁止使用未经初始化的变量

#### 2.6表达式

##### 建议2.6.1表达式的比较遵循左侧倾向于变化、右侧倾向于不变的原则

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

##### 规则2.6.2通过使用括号明确操作符的优先级，避免出现低级错误

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

#### 2.7转换

##### 规则2.7.1使用有C++提供的类型转换，而不是C风格的类型转换，避免使用const_cast和reinterpret_cast

#### 2.8控制语句

##### 规则2.8.1 switch语句要有default分支

#### 2.9声明与初始化

##### 规则2.9.1禁止用`memcpy_s`、`memset_s`初始化非POD对象

#### 2.10指针和数组

##### 规则2.10.1禁止持有std:string的c_str()返回的指针

```cpp
// 错误示范
const char * a = std::to_string(12345).c_str();
```

##### 规则2.10.2优先使用unique_ptr而不是shared_ptr

##### 规则2.10.3使用std:make_shared而不是new创建shared_ptr

```cpp
// 正确示范
std::shared_ptr<FooBar> foo = std::make_shared<FooBar>();
// 错误示范
std::shared_ptr<FooBar> foo(new FooBar());
```

##### 规则2.10.4使用智能指针管理对象，避免使用new/delete

##### 规则2.10.5禁止使用auto_ptr

##### 规则2.10.6对于指针和引用类型的形参，如果是不需要修改的，要求使用const

##### 规则2.10.7数组作为函数参数时，必须同时将其长度作为函数的参数

```cpp
int ParseMsg(BYTE *msg, size_t msgLen) {
  ...
}
```

#### 2.11字符串

##### 规则2.11.1对字符串进行存储操作，确保字符串有'\0'结束符

#### 2.12断言

##### 规则2.12.1断言不能用于校验程序在运行期间可能导致的错误，可能发生的运行错误要用错误处理代码来处理

#### 2.13类和对象

##### 规则2.13.1单个对象释放使用delete，数组对象释放使用delete[]

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

##### 规则2.13.2禁止使用std:move操作const对象

##### 规则2.13.3严格使用virtual/override/final修饰虚函数

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

#### 2.14函数设计

##### 规则2.14.1使用RAII特性来帮助追踪动态分配

```cpp
// 正确示范
{
  std::lock_guard<std::mutex> lock(mutex_);
  ...
}
```

##### 规则2.14.2非局部范围使用lambdas时，避免按引用捕获

```cpp
{
  int local_var = 1;
  auto func = [&]() { ...; std::cout << local_var << std::endl; };
  thread_pool.commit(func);
}
```

##### 规则2.14.3禁止虚函数使用缺省参数值

##### 建议2.14.4使用强类型参数\成员变量，避免使用void*

#### 2.15函数使用

##### 规则2.15.1函数传参传递，要求入参在前，出参在后

```cpp
bool Func(const std::string &in, FooBar *out1, FooBar *out2);
```

##### 规则2.15.2函数传参传递，要求入参用`const T &`，出参用`T *`

```cpp
bool Func(const std::string &in, FooBar *out1, FooBar *out2);
```

##### 规则2.15.3函数传参传递，不涉及所有权的场景，使用T *或const T &作为参数，而不是智能指针

```cpp
// 正确示范
  bool Func(const FooBar &in);
  // 错误示范
  bool Func(std::shared_ptr<FooBar> in);
```

##### 规则2.15.4函数传参传递，如需传递所有权，建议使用shared_ptr+move传参

```cpp
class Foo {
  public:
    explicit Foo(shared_ptr<T> x):x_(std::move(x)){}
  private:
    shared_ptr<T> x_;
};
```

##### 规则2.15.5单参数构造函数必须用explicit修饰，多参数构造函数禁止使用explicit修饰

```cpp
explicit Foo(int x);          //good :white_check_mark:
  explicit Foo(int x, int y=0); //good :white_check_mark:
  Foo(int x, int y=0);          //bad  :x:
  explicit Foo(int x, int y);   //bad  :x:
```

##### 规则2.15.6拷贝构造和拷贝赋值操作符应该是成对出现或者禁止

```cpp
class Foo {
  private:
    Foo(const Foo&) = default;
    Foo& operator=(const Foo&) = default;
    Foo(Foo&&) = delete;
    Foo& operator=(Foo&&) = delete;
};
```

##### 规则2.15.7禁止保存、delete指针参数

##### 规则2.15.8禁止使用非安全函数，需要给出清单

##### 规则2.15.9禁止使用非安全退出函数，需要给出清单

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

##### 规则2.15.10禁用rand函数产生用于安全用途的伪随机数

C标准库rand()函数生成的是伪随机数，请使用/dev/random生成随机数。

##### 规则2.15.11严禁使用string类存储敏感信息

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

##### 规则2.15.12内存中的敏感信息使用完毕后立即清0

口令、密钥等敏感信息使用完毕后立即清0，避免被攻击者获取。

#### 2.16内存

##### 规则2.16.1内存分配后必须判断是否成功

内存分配失败后，那么后续的操作存在未定义的行为风险。比如malloc申请失败返回了空指针，对空指针的解引用是一种未定义行为。

##### 规则2.16.2禁止引用未初始化的内存

malloc、new分配出来的内存没有被初始化为0，要确保内存被引用前是被初始化的。

##### 规则2.16.3避免使用realloc()函数

随着参数的不同，realloc函数行为也不同，这不是一个设计良好的函数。虽然在编码中提供了一些便利性，但是却极易引发各种bug。

##### 规则2.16.4不要使用alloca()函数申请栈上内存

POSIX和C99均未定义alloca()的行为，在有些平台下不支持该函数，使用alloca会降低程序的兼容性和可移植性，该函数在栈帧里申请内存，申请的大小很可能超过栈的边界，影响后续的代码执行。

#### 2.17文件

##### 规则2.17.1必须对文件路径进行规范化后再使用

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

##### 规则2.17.2不要在共享目录中创建临时文件

程序的临时文件应当是程序自身独享的，任何将自身临时文件置于共享目录的做法，将导致其他共享用户获得该程序的额外信息，产生信息泄露。因此，不要在任何共享目录创建仅由程序自身使用的临时文件。
如Linux下的/tmp目录是一个所有用户都可以访问的共享目录，不应在该目录下创建仅由程序自身使用的临时文件。

#### 2.18安全函数

| 安全函数类型 | 说明                                             | 备注                                                                                                                                                                                                                                                                                                                                                                       |
| ------------ | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| xxx_s        | Huawei Secure C库的安全函数API                   | 集成Huawei Secure C库即可使用                                                                                                                                                                                                                                                                                                                                              |
| xxx_sp       | Huawei Secure C库的安全函数性能优化API（宏实现） | 性能优化宏接口对count、destMax、strSrc为常量时有优化效果，如果是变量则优化效果不明显。宏接口使用策略：默认使用\_s接口，在性能敏感的调用点受限使用\_sp接口，受限场景如下：a) memset\_sp/memcpy\_sp使用场景：destMax和count为常量；b) strcpy\_sp/strcat\_sp使用场景：destMax为常量且strSrc为字面量；c) strncpy\_sp/strncat\_sp使用场景：destMax和count为常量且strSrc为字面量 |

##### 规则2.18.1请使用社区提供的安全函数库的安全函数，禁止使用内存操作类危险函数

| 函数类别       | 危险函数      | 安全替代函数                       |
| -------------- | ------------- | ---------------------------------- |
| 内存拷贝       | memcpy或bcopy | memcpy_s                           |
|                | wmemcpy       | wmemcpy_s                          |
|                | memmove       | memmove_s                          |
|                | wmemmove      | wmemmove_s                         |
| 字符串拷贝     | strcpy        | strcpy_s                           |
|                | wcscpy        | wcscpy_s                           |
|                | strncpy       | strncpy_s                          |
|                | wcsncpy       | wcsncpy_s                          |
| 字符串串接     | strcat        | strcat_s                           |
|                | wcscat        | wcscat_s                           |
|                | strncat       | strncat_s                          |
|                | wcsncat       | wcsncat_s                          |
| 格式化输出     | sprintf       | sprintf_s                          |
|                | swprintf      | swprintf_s                         |
|                | vsprintf      | vsprintf_s                         |
|                | vswprintf     | vswprintf_s                        |
|                | snprintf      | snprintf_s或snprintf_truncated_s   |
|                | vsnprintf     | vsnprintf_s或vsnprintf_truncated_s |
| 格式化输入     | scanf         | scanf_s                            |
|                | wscanf        | wscanf_s                           |
|                | vscanf        | vscanf_s                           |
|                | vwscanf       | vwscanf_s                          |
|                | fscanf        | fscanf_s                           |
|                | fwscanf       | fwscanf_s                          |
|                | vfscanf       | vfscanf_s                          |
|                | vfwscanf      | vfwscanf_s                         |
|                | sscanf        | sscanf_s                           |
|                | swscanf       | swscanf_s                          |
|                | vsscanf       | vsscanf_s                          |
|                | vswscanf      | vswscanf_s                         |
| 标准输入流输入 | gets          | gets_s                             |
| 内存初始化     | memset        | memset_s                           |

##### 规则2.18.2正确设置安全函数中的destMax参数

##### 规则2.18.3禁止封装安全函数

##### 规则2.18.4禁止用宏重命名安全函数

```cpp
#define XXX_memcpy_s memcpy_s
#define SEC_MEM_CPY memcpy_s
#define XX_memset_s(dst, dstMax, val, n) memset_s((dst), (dstMax), (val), (n))
```

##### 规则2.18.5禁止自定义安全函数

使用宏重命名安全函数不利于静态代码扫描工具（非编译型）定制针对安全函数误用的规则，同时，由于命名风格多
样，也不利于提示代码开发者函数的真实用途，容易造成对代码的误解及重命名安全函数的误用。重命名安全函数不
会改变安全函数本身的检查能力。

```cpp
void MemcpySafe(void *dest, unsigned int destMax, const void *src, unsigned int count) {
  ...
}
```

##### 规则2.18.6必须检查安全函数返回值，并进行正确的处理

原则上，如果使用了安全函数，需要进行返回值检查。如果返回值！=EOK，那么本函数一般情况下应该立即返回，不
能继续执行。
安全函数有多个错误返回值，如果安全函数返回失败，在本函数返回前，根据产品具体场景，可以做如下操作（执行
其中一个或多个措施）：
(1)记录日志
(2)返回错误
(3)调用abort立即退出程序

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

##### 规则2.18.7禁止外部可控数据作为system、popen、WinExec、ShellExecute、execl、xeclp、execle、execv、execvp、CreateProcess等进程启动函数的参数

##### 规则2.18.8禁止外部可控数据作为dlopen/LoadLibrary等模块加载函数的参数

##### 规则2.18.9禁止在信号处理例程中调用非异步安全函数

信号处理例程应尽可能简化。在信号处理例程中如果调用非异步安全函数，可能会导致函数的执行不符合预期的结
果。下列代码中的信号处理程序通过调用fprintf()写日志，但该函数不是异步安全函数。

```cpp
void Handler(int sigNum) {
  ...
  fprintf(stderr, "%s\n", info);
}
```

### 3. 安全编码

#### 3.1总体原则

##### 规则3.1.1保证静态类型安全

C++应该是静态类型安全的，这样可以减少运行时的错误，提升代码的健壮性。但是由于C++存在下面的特性，会破坏C++静态类型安全，针对这部分特性要仔细处理：

- 联合体
- 类型转换
- 缩窄转换
- 类型退化
- 范围错误
- void* 类型指针

可以通过约束这些特性的使用，或者使用C++的新特性，例如std:variant(C++17)、std:span(C++20)等来解决这些问题，提升C++代码的健壮性。

##### 规则3.1.2保证内存安全

C++语言的内存完全由程序员自己控制，所以在操作内存的时候必须保证内存安全，防止出现内存错误：

- 内存越界访问
- 释放以后继续访问内存
- 解引用空指针
- 内存没有初始化
- 把指向局部变量的引用或者指针传递到了函数外部或者或者其他线程中
- 申请的内存或者资源没有及时释放

建议使用更加安全的C++的特性，比如RAII、引用、智能指针等，来提升代码的健壮性。

##### 规则3.1.3禁止使用编译器"未定义行为"

遵循ISO C++标准，标准中未定义的行为禁止使用。对于编译器实现的特性或者GCC等编译器提供的扩展特性也需要谨慎使用，这些特性会降低代码的可移植性。

#### 3.2类

##### 规则3.2.1 delete操作符、移动构造函数、移动赋值操作符、swap函数应该有noexcept声明

#### 3.3表达式与语句

##### 规则3.3.1禁止逐位操作非trivially copyable对象

#### 3.4资源管理

##### 规则3.4.1 new和delete配对使用，new[]和delete[]配对使用

##### 规则3.4.2自定义new/delete操作符需要配对定义，且行为与被替换的操作符一致

##### 规则3.4.3使用恰当的方式处理new操作符的内存分配错误

##### 规则3.4.4避免出现delete this操作

#### 3.5标准库

##### 规则3.5.1禁止从空指针创建std:string

##### 规则3.5.2不要保存std:string类型的 `c_str` 和 `data` 成员函数返回的指针

##### 规则3.5.3避免使用atoi、atol、atoll、atof函数

##### 规则3.5.4禁止使用std:string存储敏感信息

##### 规则3.5.5调用格式化输入/输出函数时，禁止format参数受外部数据控制

##### 规则3.5.6禁用程序与线程的退出函数和atexit函数

##### 规则3.5.7禁止调用kill、TerminateProcess函数直接终止其他进程

---

### 4. CANN生态高频缺陷模式

本章节基于CANN生态8个仓库（hccl, hcomm-dev, hccl-dev, ops-transformer, ops-nn, ops-nn-dev, ops-transformer-dev等）共9237条提交、2279条确认缺陷的全量分析综合。收录的规则均为现有规范章节(1-3)尚未覆盖的CANN领域特有缺陷模式。

#### 4.1整数溢出与类型宽度安全

跨仓频次： ~160条 | 覆盖仓库：全部(7/7) | 严重度：P0

CANN生态最普遍的高危缺陷模式。现有规范3.1.1"保证静态类型安全"仅作原则性要求，以下规则针对CANN代码中反复出现的具体子模式做出明确约束。

##### 规则4.1.1 shape维度算术运算和GM偏移量必须使用int64_t/uint64_t

典型缺陷表现： `uint32_t offset = batch * seq * head * dim`，当batch=64, seq=4096, head=128, dim=128时乘积为274,877,906,944，超出uint32上限4,294,967,295导致静默溢出。

审查要点：
- shape维度变量（batch, seq, head, dim等）的算术运算中，至少一个操作数须先cast到int64_t/uint64_t再参与运算
- GM地址/偏移量变量统一使用uint64_t声明
- 搜索`uint32_t * uint32_t`和`int32_t * int32_t`的乘法模式

##### 规则4.1.2 DataCopy的stride/blockLen参数须校验不超过uint16上限

典型缺陷表现：NPU硬件指令DataCopy的stride/blockLen参数为16位宽度，当值超过65535时发生静默截断，导致搬运错误。

审查要点：
- DataCopy stride/count赋值前校验不超uint16上限(65535)
- 超出时须使用DataCopyExtParams或分段搬运
- 核对参数类型和值域，特别关注动态shape场景下值可能超限

##### 规则4.1.3 uint无符号减法前必须判断被减数不小于减数

典型缺陷表现： `uint32_t result = a - b`，当a < b时wrap around为极大正数（如4294967295），后续作为size/offset使用导致越界。

审查要点：
- uint减法`a - b`前必须有`a >= b`的前置判断
- 特别关注循环中的索引减法和size差值计算

##### 规则4.1.4禁止有符号与无符号整数的隐式混合比较

典型缺陷表现： `int64_t dynShape = -1`（表示动态shape）隐式转换为`uint64_t`后变为极大正数，比较和分支逻辑全部失效。

审查要点：
- 编译选项须开启`-Wsign-compare`(建议`-Werror=sign-compare`)
- 动态shape标志值(-1)不得与无符号类型变量直接比较
- 字节数/元素数不得混淆：确认API期望的单位（字节vs元素）与传入值一致

##### 规则4.1.5宽类型向窄类型赋值须显式检查值域

典型缺陷表现：u64赋给u32截断高32位；u32超时值经u8返回截断。

审查要点：
- 编译选项须开启`-Wconversion`(建议`-Werror=conversion`)
- 跨宽度赋值前须有值域范围判断或static_cast配合注释说明
- 重点关注函数返回值向窄类型变量的赋值

#### 4.2条件分支与逻辑覆盖

跨仓频次： ~220条 | 覆盖仓库：全部(7/7) | 严重度：P0-P1

现有规范2.8.1仅要求switch有default分支，以下规则覆盖CANN代码中更普遍的分支遗漏和逻辑错误模式。

##### 规则4.2.1新增枚举值/Layout/模式时须全局搜索确认所有分支覆盖

典型缺陷表现：新增NTD layout后，多处if/switch/映射表未覆盖新值，导致运行时走入错误分支或未定义行为。

审查要点：
- 新增枚举值后，全局搜索该枚举类型的所有switch/if-else链/映射表，逐一确认覆盖
- CANN配置空间维度：Layout(BNSD/BSND/TND/NTD) x模式(GQA/MHA/MQA) x量化(FP8/INT8/Perblock) x平台(910B/910_93/910_95)
- switch语句的default分支应包含错误处理（日志+返回错误）而非静默忽略

##### 规则4.2.2多枚举值否定校验中禁止用`||`连接`!=`

典型缺陷表现： `x != 3 || x != 4`恒为true（De Morgan律误用），正确写法应为`x != 3 && x != 4`。

审查要点：
- grep检查`!= ... ||`模式
- 多个`!=`用`||`连接时99%应该用`&&`
- 编译选项建议开启`-Wtautological-compare`

##### 规则4.2.3 CHECK/ASSERT宏的条件参数语义方向须与宏定义一致

典型缺陷表现：OP_CHECK_IF宏定义为"条件为true时报错返回"，但调用者传入"条件为true表示正常"的表达式，导致正常输入被拦截、非法输入被放行。

审查要点：
- 使用CHECK类宏前须确认：true表示"通过"还是"失败"
- 逐个核对OP_CHECK_IF/OP_CHECK的条件参数方向
- 校验条件`< 0`与`<= 0`：确认0是否为合法值（空tensor场景0通常合法）

##### 规则4.2.4多版本/多路径初始化分支须对称性检查

典型缺陷表现：V2适配路径遗漏了V1已有的初始化步骤；图模式路径遗漏了单算子模式已有的参数校验。

审查要点：
- V1/V2、图模式/单算子模式等多条并行路径须逐一比对，确认关键步骤（初始化/校验/清理）全覆盖
- 新增设备类型后须搜索所有设备类型分支，确认新类型已覆盖

#### 4.3流水线同步与硬件事件（算子仓特有）

跨仓频次： ~86条 | 覆盖仓库：全部算子仓(4/7) | 严重度：P0

AscendC NPU特有的高危缺陷。NPU有MTE2/MTE3/Vector/Scalar四条并行流水线，数据在不同流水线间流动需要显式同步。同步原语的正确使用需要理解底层硬件数据流，门槛高且无编译期检查。

##### 规则4.3.1 DataCopy前后须有正确方向的流水线同步

典型缺陷表现：DataCopy(MTE2)搬入数据后缺少对应管道的PipeBarrier，向量运算在数据就绪前开始执行，导致使用脏数据。

审查要点：
- DataCopy(MTE2)前确保前序MTE3完成；向量运算(V)前确保MTE2数据就绪
- PipeBarrier必须在FreeTensor之前（先确保计算完成，再释放buffer）
- 关注条件分支中某些分支是否遗漏了barrier

##### 规则4.3.2 SetFlag/WaitFlag必须成对出现且无条件执行

典型缺陷表现：条件分支中某分支只有SetFlag没有WaitFlag，导致硬件事件状态不一致；或在if-else的某个分支中遗漏SetFlag，导致后续WaitFlag永久阻塞。

审查要点：
- 全局搜索SetFlag和WaitFlag的配对关系，确认一一对应
- 配对的SetFlag/WaitFlag不得被条件分支分隔（两者须在同一控制流层级）
- 若必须条件化，则每个分支都须有完整的Set/Wait对

##### 规则4.3.3 HardEvent模板参数方向须与数据流一致

典型缺陷表现： `SyncFunc<V_MTE2>`方向反了，应为`MTE2_V`。模板参数`A_B`表示"A完成后B可以开始"。

审查要点：
- HardEvent模板参数命名约定： `A_B`表示"A流水线完成后，B流水线可以开始"
- 核对数据实际流向与模板参数方向是否一致
- 手动同步的kernel必须设置`--cce-auto-sync=off`，避免与编译器自动插入的同步冲突

#### 4.4 Host(Tiling)侧与Kernel侧一致性（算子仓特有）

跨仓频次： ~80条 | 覆盖仓库：全部算子仓(4/7) | 严重度：P0-P1

算子的Tiling-Kernel两阶段架构下，host侧(op_host/)和kernel侧(op_kernel/)代码物理分离在不同目录和编译单元，缺乏编译期一致性约束。

##### 规则4.4.1 workspace/buffer大小计算的tiling侧和kernel侧须交叉比对

典型缺陷表现：tiling用原始值计算workspace大小，kernel用对齐后的值计算，两侧不一致导致kernel侧越界访问。

审查要点：
- diff中修改workspace/buffer计算时，须同时检查tiling和kernel两侧代码
- 对齐策略（AlignUp的参数和时机）两侧须一致
- 单位（字节/元素/block）两侧须一致：AlignUp后已是字节数，不得再除sizeof

##### 规则4.4.2 TilingData结构体须使用共享头文件定义

典型缺陷表现：host侧和kernel侧各自维护TilingData结构体定义，字段数/类型/顺序不匹配，导致kernel侧解析出错误的tiling参数。

审查要点：
- TilingData结构体建议在公共头文件中定义，host和kernel两端从同一头文件引入
- 新增tiling字段时，确认两侧是否都做了对应修改
- 建议使用static_assert校验结构体size一致性

##### 规则4.4.3 tilingKey修改须同时包含host和kernel两侧的diff

典型缺陷表现：host侧生成的tilingKey值在kernel侧无法匹配，导致走入错误的kernel分支或匹配失败。

审查要点：
- tilingKey相关PR的diff必须同时包含op_host/和op_kernel/两个目录的修改
- host侧生成的key值集合与kernel侧接受的key值集合须严格一致
- TilingParse注册的类型须与tiling实际操作的类型一致

#### 4.5复制粘贴错误

跨仓频次： ~100条 | 覆盖仓库：全部(7/7) | 严重度：P1-P2

CANN代码中存在大量结构相似的代码（同族executor、同族算子、V1/V2版本、多平台分支）。复制-修改是主要开发模式，但变量名替换容易遗漏。现有规范未针对此模式做出要求。

##### 规则4.5.1新增代码与相邻已有代码高度相似时须逐行比对变量名

典型缺陷表现：从query处理代码复制到key处理，但变量名仍为query; 矩阵k_dim和n_dim公式完全相同（应不同）。

审查要点：
- diff中新增代码与相邻已有代码结构高度相似时，逐行比对变量名是否正确替换
- 函数调用中两个参数完全相同`f(a, a)`时须确认非copy-paste错误(如`isEmptyTensor(batch1, batch1)`第二参数应为batch2)
- 日志前缀/tag必须与当前函数名或类名匹配，不得残留源代码的标识

##### 规则4.5.2 CMake中的OP_NAME须与实际算子目录名一致

典型缺陷表现：从MoeFinalizeRouting复制CMake配置但OP_NAME未改为MoeInitRouting，导致算子注册名错误。

审查要点：
- CMakeLists.txt中OP_NAME变量须与所在目录的算子名一致
- CI中加入OP_NAME与目录名一致性自动校验

#### 4.6构建系统/CMake配置

跨仓频次： ~230条 | 覆盖仓库：全部(7/7) | 严重度：P1-P2

CANN生态中频次最高的缺陷类别。CMake构建系统的复杂度随算子数量和平台数量超线性增长。

##### 规则4.6.1新增算子PR须包含完整的构建配置修改

典型缺陷表现：新增算子的.cc文件已编写但未在CMakeLists.txt中注册；ascendc_config.json/binary.json遗漏导致运行时kernel not found。

审查要点：
- 新增算子PR必须包含：CMakeLists.txt + ascendc_config.json + binary.json + operator_list.yaml的对应修改
- PR文件列表中缺少上述任一配置文件须视为不完整

##### 规则4.6.2新平台适配须全局搜索确认所有编译分支覆盖

典型缺陷表现：新增ascend910_95时CMake条件分支未覆盖，导致新平台编译失败或链接到错误的库。

审查要点：
- 新平台适配PR须全局搜索`ASCEND_COMPUTE_UNIT`/`socVersion`/`__CCE_AICORE__`所有出现位置，逐一确认覆盖
- CI门禁应覆盖所有编译目标(host/device/kernel/daemon)的构建验证
- shell脚本须通过shellcheck验证

#### 4.7空指针与初始化

跨仓频次： ~70条 | 覆盖仓库：6/7 | 严重度：P0-P1

现有规范2.5.5"禁止使用未经初始化的变量"和3.1.2"保证内存安全"已作原则性要求，以下规则针对CANN代码中反复出现的具体子模式做出补充约束。

##### 规则4.7.1 C++类的内置类型成员须有in-class initializer

典型缺陷表现：bool/uint32_t等成员变量无in-class initializer，构造函数遗漏初始化时成员为随机值，后续分支逻辑不确定。

审查要点：
- 类定义中所有内置类型（bool, int, uint32_t, 指针等）成员须有`= 0`/`= false`/`= nullptr`等in-class初始化
- 特别关注新增成员变量是否遗漏初始化

##### 规则4.7.2指针解引用（含作为函数参数）须在nullptr检查之后

典型缺陷表现： `val = ptr->Get(); if (ptr == nullptr) { ... }`先解引用后判空，判空代码形同虚设。

审查要点：
- 搜索"先用后查"模式：指针解引用出现在nullptr检查之前
- 可选输入/输出tensor访问前须有null/flag守卫（如outputMask控制的tensor）
- nullptr不得传给%s格式化参数（导致崩溃）

##### 规则4.7.3 OP_CHECK宏的返回参数须包含return语句

典型缺陷表现： `OP_CHECK(cond, msg, nullptr)`应为`return nullptr`，缺少return导致CHECK失败后继续执行后续代码。

审查要点：
- OP_CHECK/OP_CHECK_IF宏的第三参数（或失败处理参数）须包含return
- graphStatus函数中return bool值为语义错误（应return graphStatus枚举值）

#### 4.8通信库特有缺陷模式

以下模式主要出现在通信库(hccl/hcomm-dev)，反映多线程/多设备/多版本环境下的状态管理和同步挑战。算子仓库中几乎不出现。

##### 规则4.8.1 [P0]并发数据发布须有memory fence

典型缺陷表现："先写数据后更新标志"缺memory fence，消费者线程可能看到更新后的标志但读到旧数据（指令重排序/缓存可见性）。

审查要点：
- 生产者-消费者模式中，数据写入与标志更新之间须有memory barrier(`std::atomic`的release/acquire或显式fence)
- 搜索"先写payload后置flag"的代码模式

##### 规则4.8.2 [P0]并发的delete/destroy操作须加锁保护

典型缺陷表现：两个线程同时delete同一对象导致double-free; 析构函数中释放资源的顺序与组件间依赖关系不一致导致UAF。

审查要点：
- 搜索delete后判nullptr模式(`delete ptr; ptr = nullptr`非线程安全)
- 资源释放顺序须与组件间依赖关系的逆序一致
- 建议使用RAII和智能指针管理共享资源的生命周期

##### 规则4.8.3 [P0]协议OpCode/数据结构修改须保持版本兼容

典型缺陷表现：修改已发布OpCode的数据结构但编号不变，新旧版本节点解析同一OpCode时字段错位。

审查要点：
- 修改已发布OpCode对应的数据结构时须同步修改OpCode编号（或增加新OpCode）
- 须考虑滚动升级场景：新旧版本节点可能同时存在

##### 规则4.8.4 [P1]缓存复用时运行时字段须刷新

典型缺陷表现：cache key维度不足导致错误命中；缓存命中后复用对象的运行时字段（如stream）未刷新，使用了过期的stream句柄。

审查要点：
- cache key须包含所有影响对象状态的维度
- cache命中路径中须检查运行时字段（stream、device_id等）是否需要刷新
- thread_local变量在新线程中须确保经过设置路径

##### 规则4.8.5 [P1]单例/引用计数初始化须区分首次与后续调用

典型缺陷表现：引用计数 > 1时跳过整个Init函数，但Init中包含每次调用都需要执行的通信域配置步骤，导致后续通信域缺少必要配置。

审查要点：
- Init函数中须区分"仅首次执行"和"每次调用都执行"的逻辑
- 引用计数跳过的范围须精确控制，不得跳过每次调用都需要的配置步骤

#### 4.9算子仓特有缺陷模式

以下模式主要出现在算子仓库（ops-transformer/ops-nn系列），涉及算子特有的计算语义和AscendC编程模型。

##### 规则4.9.1 [P0] GQA场景须检查gSize缩放因子

典型缺陷表现：Q head数是KV head数的gSize倍，多处交叉计算（如preTokens/nextTokens与head维度相关的运算）遗漏gSize因子，导致Attention计算错误。

审查要点：
- 涉及Q/KV head数交叉计算的代码须确认gSize因子是否正确使用
- 搜索preTokens、nextTokens等变量，确认关联维度的缩放关系

##### 规则4.9.2 [P1]空tensor须全链路处理

典型缺陷表现：aclnn层允许空tensor(dim=0)输入，但infershape/tiling/kernel三层未联动处理，导致后续除零或越界。

审查要点：
- 空tensor（任一维度为0）须在aclnn/infershape/tiling/kernel四层全部有对应处理
- 校验条件`< 0`与`<= 0`须区分：0通常是空tensor的合法值，不应被拦截

##### 规则4.9.3 [P1] CeilDiv与CeilAlign语义禁止混淆

典型缺陷表现：CeilDiv返回块数，CeilAlign返回对齐后的字节数/元素数，两者差n倍(n=blockSize)。混用导致buffer大小计算错误。

审查要点：
- CeilDiv(x, n) = 向上取整的块数
- CeilAlign(x, n) = 向上对齐后的值 = CeilDiv(x, n) * n
- 审查中须确认调用方期望的语义与实际调用的函数是否匹配

##### 规则4.9.4 [P1] StorageShape与ViewShape禁止混淆

典型缺陷表现：非连续tensor的物理shape(StorageShape)与逻辑shape(ViewShape)不同，使用错误的shape计算偏移导致数据错位。

审查要点：
- 涉及非连续tensor（stride不为连续排列）的代码须确认使用的是StorageShape还是ViewShape
- 偏移计算须使用stride而非简单的shape连乘

##### 规则4.9.5 [P1]对齐计算须保证方向正确且满足硬件约束

典型缺陷表现：不必要的对齐破坏地址计算（如对已对齐值重复对齐改变了语义）；未满足NPU要求的32B对齐导致硬件异常。

审查要点：
- NPU数据搬运通常要求32B对齐，确认AlignUp的对齐参数是否满足硬件约束
- 对齐操作的位置须在正确的计算阶段（过早对齐可能改变后续算术语义）
- 确认对齐后的值的单位与后续使用处的期望单位一致

#### 4.10大规模提交与流程规范

跨仓频次： ~30个独立事件 | 覆盖仓库：6/7 | 严重度：P2

##### 规则4.10.1单次合入超过3000行或20个文件的PR须拆分

典型缺陷表现：6.5万行一次性合入导致3次循环revert; tilingKey模板化方案5天内6次Revert。

审查要点：
- 大体量PR须按功能维度拆分为独立可验证的子PR
- 架构方案在首个模块失败后应暂停，完成根因分析再继续推广

##### 规则4.10.2功能变更不得混入批量同步/清理提交

典型缺陷表现：MR中混入不相关改动，revert时不相关改动被一并回滚（附带伤害）；commit message为"update"但实际包含678个文件的结构性变更。

审查要点：
- 功能变更不得与批量格式化/同步/清理操作混在同一PR
- 公共文件（op_util.h, error_util.h, cmake等）修改须独立PR先行合入验证

##### 规则4.10.3接口/API变更须全量搜索调用点并同步修改

典型缺陷表现：修改Init函数签名后仅修改了部分调用点，遗漏的调用点编译报错或行为异常。

审查要点：
- 修改函数签名/接口定义时，须全量搜索所有调用点并同步修改
- 禁止依赖未发布的SDK接口（抢跑依赖）

#### 4.11编译选项建议

以下编译选项可自动拦截本章节中约15%的缺陷模式（变量遮蔽、类型截断、格式串不匹配、符号比较、恒真表达式），建议在CMake中统一启用：

```
-Werror -Wshadow -Wconversion -Wformat -Wsign-compare -Wtautological-compare
```

建议在CI中集成的自动化检查：
- shellcheck: 拦截shell脚本语法错误
- ascendc_config.json重复条目和算子目录交叉校验
- SetFlag/WaitFlag配对检测（静态分析脚本）
- CMake OP_NAME与目录名一致性验证
- Check函数返回值丢弃检测(配合`[[nodiscard]]`和clang-tidy)
