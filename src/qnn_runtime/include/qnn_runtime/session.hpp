#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace qnn_runtime {
enum class Type { Float32, Float16, Int32, UInt32, Bool8, UInt8 };
enum class Role { Input, Output, Native, Static };
using Shape = std::vector<std::uint32_t>;
struct Tensor {
    std::string name;
    Type type;
    Shape shape;
    std::vector<std::uint8_t> bytes;
    std::size_t elements() const;
};
struct Param {
    std::string name;
    Type type;
    Shape shape; // empty: scalar
    std::vector<std::uint8_t> bytes;
    static Param f32(std::string name, float value);
    static Param u32(std::string name, std::uint32_t value);
    static Param boolean(std::string name, bool value);
    static Param u32s(std::string name, std::vector<std::uint32_t> values, Shape shape = {});
    static Param i32s(std::string name, std::vector<std::int32_t> values, Shape shape = {});
};
class Session {
public:
    explicit Session(const std::string& runtime, bool profile = false);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    void load(const std::string& model);
    void create(const std::string& graph_name);
    void add_tensor(std::string name, Type type, Shape shape, Role role,
                    std::span<const std::uint8_t> data = {});
    void op(std::string name, std::string type, std::vector<std::string> inputs,
            std::vector<std::string> outputs, std::vector<Param> params = {});
    void finalize();
    void run();
    void save(const std::string& path);
    void write_profile(const std::string& path) const;
    Tensor& tensor(const std::string& name);
    const Tensor& tensor(const std::string& name) const;
    std::vector<std::string> inputs() const;
    std::vector<std::string> outputs() const;
    void write_f32(const std::string& name, std::span<const float> data);
    std::vector<float> read_f32(const std::string& name) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
std::size_t element_size(Type type);
std::string shape_string(const Shape& shape);
} // namespace qnn_runtime
