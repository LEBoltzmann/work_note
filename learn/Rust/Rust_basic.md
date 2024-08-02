# Rust入门
[TOC]
本文档基于Rust语言圣经学习Rust语言。
# Cargo入门
## 创建项目
Rust使用cargo管理项目，在创建项目时使用：
```
cargo new FILENAME
cd FILENAME
```
cargo会生成项目目录和文件。编译运行时：
```
cargo run
```
`cargo run`包含了编译运行的步骤，如果只编译可以使用`cargo build`。另外这两个命令都是debug模式，为了在开发阶段快速编译，所以运行慢。如果要提高性能运行：
```
cargo run --release
cargo build --release
```
## cargo check
`cargo check`可以快速检查代码是否能编译通过，在大型项目中很好用。
## cargo.toml和cargo.lock
* cargo.toml是项目数据描述文件，储存了所有元配置信息。
* cargo.lock是项目依赖清单，一般根据cargo.toml生成，不需要修改。

#### package配置
package中记录了项目描述信息：
```
[package]
name = "world_hello"
version = "0.1.0"
edition = "2021"
```
`name`是项目名称，`version`是项目版本，`edition`是项目大版本。
#### 项目依赖
在`cargo.toml`中有三种方式定义依赖：
* 基于Rust官方仓库`create.io`，通过项目版本引入。
* 基于项目源代码的git url
* 本地项目的路径
```
[dependencies]
rand = "0.3"
hammer = { version = "0.5.0"}
color = { git = "https://github.com/bjz/color-rs" }
geometry = { path = "crates/geometry" }
```
# Rust基础入门
## 变量
Rust需要手动声明变量可变，这为变量提供了安全性和灵活性。
#### 变量绑定
Rust的变量绑定语句为：`let a = "abc"`。Rust语言遵循所有权原则，每一个内存都有一个变量作为主人。

#### 变量可变性
变量声明前面加`mut`关键字可以声明可变变量。
#### 使用下划线忽略未使用变量
在Rust中创建但不使用变量会被警告，可以使使用下划线来让Rust忽略不使用的变量。使用下划线开头的变量即使在后面使用也不会报错。

```rust
fn main() {
    let _x = 5;
    let y = 10;		\\警告
	println!("{}", _x);	\\不会报错
}
```
#### 变量解构
`let`可以从较复杂的数据中解构出匹配的部分：
```rust
fn main() {
    let (a, mut b): (bool,bool) = (true, false);
    // a = true,不可变; b = false，可变
    println!("a = {:?}, b = {:?}", a, b);

    b = true;
    assert_eq!(a, b);
}
```
###### 解构式赋值
在Rust1.59版本后可以在左式中使用切片，元组和结构体：
```Rust
struct Struct {
    e: i32
}

fn main() {
    let (a, b, c, d, e);

    (a, b) = (1, 2);
    // _ 代表匹配一个值，但是我们不关心具体的值是什么，因此没有使用一个变量名而是使用了 _
    [c, .., d, _] = [1, 2, 3, 4, 5];
    Struct { e, .. } = Struct { e: 5 };

    assert_eq!([1, 2, 1, 4, 5], [a, b, c, d, e]);
}
```
!小问题：在结构体中定义但没有使用的变量也会警告，即使它只是作为右值传递。

```Rust
fn main() {
    let a = Struct { e: 1, f: 2 };
    println!("{}", a.e);

	Struct {e, ..} = Struct{e: 1, f: 2}	#都会警告
}
```
#### 变量和常量的差异
常量在编译完成就只有固定值，所以不能用`mut`关键字。常量使用`const`关键字声明：

```Rust
const LIGHT_SPEED = 299792458
```
#### 变量遮蔽（shadowing）
Rust允许声明相同的变量名，后面的会遮蔽前面的：

```Rust
fn main() {
    let x = 5;
	# 在main函数的作用域内对之前的x进行遮蔽
    let x = x + 1;

    {
	# 在当前的花括号作用域内，对之前的x进行遮蔽
        let x = x * 2;
        println!("The value of x in the inner scope is: {}", x);
    }

    println!("The value of x is: {}", x);
}
```
运行后输出12和6.

Tips：变量遮蔽不会释放前一个变量的内存，而是在生命周期结束后被释放。所以一般可以在另一个作用域内使用来遮蔽掉作用域外的变量。

## 基本类型
Rust类型氛围基本类型和复合类型，复合类型一般来说无法解构。基本类型包括：

* 数值类型，包括有符号整数（`i8, i16, i32, i64, isize`），无符号整数（`u8, u16, u32, u64, usize`），浮点数（`f32, f64`）以及有理数、复数。
* 字符串型，包括字符串字面量和字符串切片（`&str`）
* 布尔型，包括 `true, false`
* 字符类型，包括单个unicode字符。大小为4字节。
* 单元类型，即`()`。唯一的值也是`()`。


### 类型推导与标注
Rust是静态语言，需要在编译阶段知道数据类型。但是Rust可以通过上下文推导数据类型，有时Rust也无法推导出类型时会报错：

```Rust
let guess = "42".parse().expect("not a number!");	#报错

let guess:i32 = "42".parse().expect("not a number!");
let guess = "42".parse()::<i32>.expect("not a number!");	#可以
```
### 数值类型
#### 整数类型
对于指定长度的整型，它的取值范围为$2^{n-1}$ ~ $2^{n-1}-1$，对于无符号是$0$ ~ $2^n -1$。isize与usize与CPU架构有关，如果CPU为32/64位即为32/64bit。
##### 整型溢出
整型溢出在编译时会崩溃，但在`--release`下会按补码循环溢出求值。但这种行为仍然应该认为是错误代码。

我们可以使用标准库中的方法处理溢出：
* `wrapping_add`按补码循环处理。
* `checked_add`溢出时返回None。
* `overflowing_add`溢出时返回补码循环值和一个是否溢出的布尔值
* `saturating_add`限定不能超过该类型最大值和最小值。

Tips: None是Option枚举的值，表示可能不存在意义的值。而()通常表示函数没有返回值，虽然意义相似但是场景不同：
```Rust
enum Option<T> {
    Some(T),
    None,
}

fn example_function() -> () {
    // 函数体
}
```
#### 浮点数
浮点数分为`f32` 和 `f64`。因为现在计算机处理两种类型的数据速度相当来，所以默认使用`f64`。

##### 浮点数陷阱
浮点数由于实现与整型不同，需要在使用时小心：

* **浮点数往往是想要数值的近似表达**：浮点数由于使用二进制精度，而我们通常使用十进制表达浮点数。比如`0.1`就不会得到精确数值。
* **浮点数在某些特性上是反直觉的**：浮点数并没有实现等于的接口，所以在某些使用上要小心。
* 要避免对浮点数使用相等判断，结果在数学上未定义时要小心。

如果一定要取等，可以选择设定最小值：

```Rust
(0.1_f64 + 0.2 - 0.3).abs() < 0.00001
```
##### NaN
数学上未定义的操作会返回NaN（not a number）。所有与NaN进行操作都会返回NaN，并且NaN不能比较。如果想要确定是否是NaN可以使用`is_nan()`方法。
#### 数字运算
Rust支持所有加减乘除模运算。
```Rust
fn main() {
  // 编译器会进行自动推导，给予twenty i32的类型
  let twenty = 20;
  // 类型标注
  let twenty_one: i32 = 21;
  // 通过类型后缀的方式进行类型标注：22是i32类型
  let twenty_two = 22i32;

  // 只有同样类型，才能运算
  let addition = twenty + twenty_one + twenty_two;
  println!("{} + {} + {} = {}", twenty, twenty_one, twenty_two, addition);

  // 对于较长的数字，可以用_进行分割，提升可读性
  let one_million: i64 = 1_000_000;
  println!("{}", one_million.pow(2));

  // 定义一个f32数组，其中42.0会自动被推导为f32类型
  let forty_twos = [
    42.0,
    42f32,
    42.0_f32,
  ];

  // 打印数组中第一个值，并控制小数位为2位
  println!("{:.2}", forty_twos[0]);
}
```
#### 位运算
Rust的位运算有：`&`按位与`|`按位或`^`按位异或`！`按位非`<<`左移填充零`>>`右移填充0，负数填充1。
#### 序列（Range）
Rust可以简洁地实现连续序列：
```Rust
for i in 1..=5 {
    println!("{}",i);
}
```
默认不包含右界，需要包含时使用`=`。Range只能用于整型和字符型。

#### 使用As类型转换
Rust使用As进行类型转换
```Rust
let a:i32 = 10;
let b:f32 = a as f32;
```

#### 有理数和复数
可以使用num库实现复数，在`cargo.toml`中添加`num = "0.4.0"`。
```Rust
use num::complex::Complex;

 fn main() {
   let a = Complex { re: 2.1, im: -1.2 };
   let b = Complex::new(11.1, 22.2);
   let result = a + b;

   println!("{} + {}i", result.re, result.im)
 }
```

### 字符、布尔、单元类型
#### 字符
Rust中字符可以使用unicode，大小为4字节。
#### bool
布尔只有`true`和`false`值，大小为1字节。
#### 单元类型
用于表示没有返回值，或者用()当作键值对中的map值来表达不关系map只关心key。单元类型不占用任何内存。

### 语句与表达式
在Rust中语句没有返回值，表达式有返回值，需要区分。
#### 语句
`let`是一个语句，没有返回值。所以不能把它赋给其他值。

Tips：我目前的版本let编译与语言圣经给出的不同：
```Terminal
error: expected expression, found `let` statement
 --> src/main.rs:3:10
  |
3 | let b = (let a = 8);
  |          ^^^
  |
  = note: only supported directly in conditions of `if` and `while` expressions

error: could not compile `playground` (bin "playground") due to 1 previous error
```

经搜索let可以与`if`、`while`结合，这是Rust中文手册给出的例子：
```Rust

let dish = ("Ham", "Eggs");

// 此主体代码将被跳过，因为该模式被反驳
if let ("Bacon", b) = dish {
    println!("Bacon is served with {}", b);
} else {
    // 这个块将被执行。
    println!("No bacon will be served");
}
```
使用时可以通过匹配来判断等式两边是否可以匹配并选择执行块。但这里仍然还是认为`let`只是与`if`、`while`配合使用而不是作为语句返回值。虽然在vscode中尝试把语句赋值给变量会认定变量为布尔值。
#### 表达式
表达式可以求值并返回一个值。在语句中`let a = 6`中的6也是一个表达式并返回值`6`。只要是可以返回值的都是表达式。

```Rust
fn main() {
    let y = {
        let x = 3;
        x + 1
    };

    println!("The value of y is: {}", y);
}
```
这里面`let y`语句后面的花括号也是表达式。表达式后面不能带分号，否则就会变成语句。也可以使用if语句来选择块表达式：
```Rust
let z = if x % 2 == 1 { "odd" } else { "even" };
```
Tips：赋值表达式会返回`()`。我在运行`println!`的时候还遇到了这样的问题：
```Rust
println!("a is {:?}", a = a+1); //输出11，不会改变外部的a
println!("a is {:?}", a += 1);  //输出()，且会改变外部a

```
这部分应该涉及到借用，上面的打印借用了a，下面的表达式只是把`a+=1`的返回值发给了输出函数。

### 函数
一个加法函数：
```Rust
fn add(i: i32, j: i32) -> i32 {
   i + j
 }
```
#### 函数要点
* 使用蛇形命名
* Rust不关心声明位置
* 参数要声明类型

#### Rust返回值
Rust通过`->`指定返回类型。因为函数也是表达式，可以在函数块结尾加上表达式来作为返回值。也可以直接使用return提前返回值。
```Rust
fn plus_or_minus(x:i32) -> i32 {
    if x > 5 {
        return x - 5
    }

    x + 5
}

fn main() {
    let x = plus_or_minus(5);

    println!("The value of x is: {}", x);
}
```
return表达式也不带分号
##### 特殊的返回值
无返回值，可以显式或隐式地说明，使用`-> ()`或者不再块最后加上表达式。
##### 发散函数
有些函数永远不返回，使用`-> !`来声明。这种返回类型经常用作会导致程序崩溃的函数。

Tips：代码圣经中习题的例子，`unimplemented!()`表示某个功能没有实现并且立即崩溃。`panic!()`用于导致恐慌，运行到这里也会直接崩溃。`todo!()`也代表未实现功能并且计划实现。

## 所有权和借用
### 所有权
目前的三种内存管理机制：
* 垃圾回收机制（GC）比如Java，Go
* 手动内存分配，比如C++
* 所有权管理内存，在编译是根据一系列规则检查。

所有权管理只在编译期进行，所以不会在运行时检查，安全又高效。
#### 一段不安全的代码
```c++
int* foo() {
    int a;          // 变量a的作用域开始
    a = 100;
    char *c = "xyz";   // 变量c的作用域开始
    return &a;
}                   // 变量a和c的作用域结束

```
这段代码由于a的内存被释放，返回的a值会报错。*c则会一直在内存上知道程序结束。这些都是内存不安全行为。
#### 栈(Stack)与堆(Heap)
了解堆栈概念对理解如何优化Rust代码性能十分重要。
##### 栈
栈按照后进先出的方式组织值，
