#include "MiniAPLJIT.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;
using namespace std;

class ASTNode;

// -------------------------------------------------
// Miscellaneous helper functions
// -------------------------------------------------

std::unique_ptr<ASTNode> LogError(const char* Str)
{
    fprintf(stderr, "Error: %s\n", Str);
    return nullptr;
}

static inline string str(const int i)
{
    return to_string(i);
}

bool is_int(const std::string& str)
{ // check with regex (does not accept leading zeroes before first digit)
    static constexpr auto max_digits = std::numeric_limits<int>::digits10;
    static const std::string ub = std::to_string(max_digits - 1);
    static const std::regex int_re("^\\s*([+-]?[1-9]\\d{0," + ub + "}|0)\\s*$");

    return std::regex_match(str, int_re);
}

// -------------------------------------------------
// Type information for MiniAPL programs
// -------------------------------------------------

enum ExprType {
    EXPR_TYPE_SCALAR,
    EXPR_TYPE_FUNCALL,
    EXPR_TYPE_VARIABLE
};

class MiniAPLArrayType {
public:
    vector<int> dimensions;

    int Cardinality()
    {
        int C = 1;
        for (auto D : dimensions) {
            C *= D;
        }
        return C;
    }

    int length(const int dim)
    {
        return dimensions.at(dim);
    }

    int dimension()
    {
        return dimensions.size();
    }
};

std::ostream& operator<<(std::ostream& out, MiniAPLArrayType& tp)
{
    out << "[";
    int i = 0;
    for (auto T : tp.dimensions) {
        out << T;
        if (i < (int)(tp.dimensions.size() - 1)) {
            out << ", ";
        }
        i++;
    }
    out << "]";
    return out;
}

// -------------------------------------------------
// AST classes
// -------------------------------------------------

// The base class for all expression nodes.
class ASTNode {
public:
    virtual ~ASTNode() = default;

    virtual Value* codegen(Function* F) = 0;
    virtual ExprType GetType() = 0;
    virtual void Print(std::ostream& out)
    {
    }
};

std::ostream& operator<<(std::ostream& out, ASTNode& tp)
{
    tp.Print(out);
    return out;
}

class StmtAST : public ASTNode {
public:
    virtual bool IsAssign() = 0;
};

class ProgramAST : public ASTNode {
public:
    std::vector<unique_ptr<StmtAST>> Stmts;
    Value* codegen(Function* F) override;
    virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }
};

class ExprStmtAST : public StmtAST {
public:
    std::unique_ptr<ASTNode> Val;

    bool IsAssign() override { return false; }
    ExprStmtAST(std::unique_ptr<ASTNode> Val_)
        : Val(std::move(Val_))
    {
    }
    Value* codegen(Function* F) override;
    virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }

    virtual void Print(std::ostream& out) override
    {
        Val->Print(out);
    }
};

class VariableASTNode : public ASTNode {

public:
    std::string Name;
    VariableASTNode(const std::string& Name)
        : Name(Name)
    {
    }

    Value* codegen(Function* F) override;

    virtual ExprType GetType() override { return EXPR_TYPE_VARIABLE; }

    virtual void Print(std::ostream& out) override
    {
        out << Name;
    }
};

class AssignStmtAST : public StmtAST {
public:
    std::unique_ptr<VariableASTNode> Name;
    std::unique_ptr<ASTNode> RHS;

    bool IsAssign() override { return true; }
    Value* codegen(Function* F) override;

    std::string GetName() const { return Name->Name; }

    AssignStmtAST(const std::string& Name_, std::unique_ptr<ASTNode> val_)
        : Name(new VariableASTNode(Name_))
        , RHS(std::move(val_))
    {
    }
    virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }
    virtual void Print(std::ostream& out) override
    {
        out << "assign ";
        Name->Print(out);
        out << " = ";
        RHS->Print(out);
    }
};

class NumberASTNode : public ASTNode {
public:
    int Val;
    NumberASTNode(int Val)
        : Val(Val)
    {
    }

    Value* codegen(Function* F) override;

    virtual ExprType GetType() override { return EXPR_TYPE_SCALAR; }

    virtual void Print(std::ostream& out) override
    {
        out << Val;
    }
};

class CallASTNode : public ASTNode {

public:
    std::string Callee;
    std::vector<std::unique_ptr<ASTNode>> Args;
    CallASTNode(const std::string& Callee,
        std::vector<std::unique_ptr<ASTNode>> Args)
        : Callee(Callee)
        , Args(std::move(Args))
    {
    }

    Value* codegen(Function* F) override;
    virtual ExprType GetType() override { return EXPR_TYPE_FUNCALL; }
    virtual void Print(std::ostream& out) override
    {
        out << Callee << "(";
        for (int i = 0; i < (int)Args.size(); i++) {
            Args.at(i)->Print(out);
            if (i < (int)Args.size() - 1) {
                out << ", ";
            }
        }
        out << ")";
    }
};

// ---------------------------------------------------------------------------
// Some global variables used in parsing, type-checking, and code generation.
// ---------------------------------------------------------------------------
static map<ASTNode*, MiniAPLArrayType> TypeTable;
static map<string, Value*> ValueTable;
static LLVMContext TheContext;
// NOTE: You will probably want to use the Builder in the "codegen" methods
static IRBuilder<> Builder(TheContext);
static std::unique_ptr<Module> TheModule;
static std::map<std::string, Value*> NamedValues;
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<MiniAPLJIT> TheJIT;

// ---------------------------------------------------------------------------
// LLVM codegen helpers
// ---------------------------------------------------------------------------
IntegerType* intTy(const int width)
{
    return IntegerType::get(TheContext, 32);
}

ConstantInt* intConst(const int width, const int i)
{
    ConstantInt* const_int32 = ConstantInt::get(TheContext, APInt(width, StringRef(str(i)), 10));
    return const_int32;
}

static void InitializeModuleAndPassManager()
{
    // Open a new module.
    TheModule->setDataLayout(TheJIT->getTargetMachine().createDataLayout());

    // Create a new pass manager attached to it.
    TheFPM = llvm::make_unique<legacy::FunctionPassManager>(TheModule.get());

    // Do simple "peephole" optimizations and bit-twiddling optzns.
    TheFPM->add(createInstructionCombiningPass());
    // Reassociate expressions.
    TheFPM->add(createReassociatePass());
    // Eliminate Common SubExpressions.
    TheFPM->add(createGVNPass());
    // Simplify the control flow graph (deleting unreachable blocks, etc).
    TheFPM->add(createCFGSimplificationPass());

    TheFPM->doInitialization();
}

// NOTE: This utility function generates LLVM IR to print out the string "to_print"
void kprintf_str(Module* mod, BasicBlock* bb, const std::string& to_print)
{
    Function* func_printf = mod->getFunction("printf");
    if (!func_printf) {
        PointerType::get(IntegerType::get(mod->getContext(), 8), 0);
        FunctionType* FuncTy9 = FunctionType::get(IntegerType::get(mod->getContext(), 32), true);

        func_printf = Function::Create(FuncTy9, GlobalValue::ExternalLinkage, "printf", mod);
        func_printf->setCallingConv(CallingConv::C);
    }

    IRBuilder<> builder(TheContext);
    builder.SetInsertPoint(bb);

    Value* str = builder.CreateGlobalStringPtr(to_print);

    std::vector<Value*> int32_call_params;
    int32_call_params.push_back(str);

    CallInst::Create(func_printf, int32_call_params, "call", bb);
}

Value* gen_pow(Module* mod, BasicBlock* bb, Value* base, Value* power)
{
    Function* func_pow = mod->getFunction("pow");
    if (!func_pow) {
        PointerType::get(IntegerType::get(mod->getContext(), 8), 0);
        FunctionType* FuncTy9 = FunctionType::get(IntegerType::get(mod->getContext(), 32), true);

        func_pow = Function::Create(FuncTy9, GlobalValue::ExternalLinkage, "pow", mod);
        func_pow->setCallingConv(CallingConv::C);
    }

    IRBuilder<> builder(TheContext);
    builder.SetInsertPoint(bb);

    std::vector<Value*> int32_call_params { base, power };

    return Builder.CreateCall(func_pow, int32_call_params);
}

// NOTE: This utility function generates code that prints out the 32 bit input "val" when
// executed.
void kprintf_val(Module* mod, BasicBlock* bb, Value* val)
{
    Function* func_printf = mod->getFunction("printf");
    if (!func_printf) {
        PointerType::get(IntegerType::get(mod->getContext(), 8), 0);
        FunctionType* FuncTy9 = FunctionType::get(IntegerType::get(mod->getContext(), 32), true);

        func_printf = Function::Create(FuncTy9, GlobalValue::ExternalLinkage, "printf", mod);
        func_printf->setCallingConv(CallingConv::C);
    }

    IRBuilder<> builder(TheContext);
    builder.SetInsertPoint(bb);

    Value* str = builder.CreateGlobalStringPtr("%d");

    std::vector<Value*> int32_call_params;
    int32_call_params.push_back(str);
    int32_call_params.push_back(val);

    CallInst::Create(func_printf, int32_call_params, "call", bb);
}

Value* LogErrorV(const char* Str)
{
    LogError(Str);
    return nullptr;
}

// ---------------------------------------------------------------------------
// Code generation functions that you should fill in for this assignment
// ---------------------------------------------------------------------------

void print_array(std::vector<int>& dimensions, int level, Value* array, int& curr_array_idx)
{
    if (level == (int)dimensions.size()) {
        std::vector<Value*> gep_indices { intConst(32, 0), intConst(32, curr_array_idx) };
        auto element_ptr = Builder.CreateGEP(array, gep_indices);

        Value* val = Builder.CreateLoad(element_ptr);
        kprintf_val(TheModule.get(), Builder.GetInsertBlock(), val);

        curr_array_idx++;
        return;
    }

    for (int i = 0; i < dimensions[level]; i++) {
        kprintf_str(TheModule.get(), Builder.GetInsertBlock(), "[");
        print_array(dimensions, level + 1, array, curr_array_idx);
        kprintf_str(TheModule.get(), Builder.GetInsertBlock(), "]");
    }
}

Value* ProgramAST::codegen(Function* F)
{
    // STUDENTS: FILL IN THIS FUNCTION
    Value* ret;
    for (unique_ptr<StmtAST>& statement : Stmts) {
        ret = statement->codegen(F);
    }

    StmtAST* last_stmt = Stmts[Stmts.size() - 1].get();
    ASTNode* node;

    if (last_stmt->IsAssign()) {
        AssignStmtAST* Assign = static_cast<AssignStmtAST*>(last_stmt);
        node = Assign->Name.get();
    } else {
        ExprStmtAST* Expr = static_cast<ExprStmtAST*>(last_stmt);
        node = Expr->Val.get();
    }
    MiniAPLArrayType type = TypeTable[node];

    int curr_array_idx = 0;
    kprintf_str(TheModule.get(), Builder.GetInsertBlock(), "[");
    print_array(type.dimensions, 0, ret, curr_array_idx);
    kprintf_str(TheModule.get(), Builder.GetInsertBlock(), "]");
    kprintf_str(TheModule.get(), Builder.GetInsertBlock(), "\n");

    Builder.CreateRet(nullptr);
    return nullptr;
}

Value* AssignStmtAST::codegen(Function* F)
{
    // STUDENTS: FILL IN THIS FUNCTION
    Value* right = RHS->codegen(F);
    NamedValues[GetName()] = right;
    return nullptr;
}

Value* ExprStmtAST::codegen(Function* F)
{
    // STUDENTS: FILL IN THIS FUNCTION
    return Val->codegen(F);
}

Value* NumberASTNode::codegen(Function* F)
{
    // STUDENTS: FILL IN THIS FUNCTION
    return intConst(32, Val);
}

Value* VariableASTNode::codegen(Function* F)
{
    // STUDENTS: FILL IN THIS FUNCTION
    Value* V = NamedValues[Name];
    if (!V)
        LogErrorV("Unknown variable name");
    return V;
}

void map_over_array(std::vector<int>& dimensions, int level, std::vector<int>& curr_indices, Value* alloc, int& curr_array_idx, std::function<llvm::Value*(int, std::vector<Value*>)> get_val_to_store)
{
    if (level == (int)dimensions.size()) {
        std::vector<Value*> gep_indices { intConst(32, 0), intConst(32, curr_array_idx) };

        Value* to_store = get_val_to_store(curr_array_idx, gep_indices);

        auto element_ptr = Builder.CreateGEP(alloc, gep_indices);
        Builder.CreateStore(to_store, element_ptr);

        curr_array_idx++;
        return;
    }

    for (int i = 0; i < dimensions[level]; i++) {
        curr_indices[level] = i;
        map_over_array(dimensions, level + 1, curr_indices, alloc, curr_array_idx, get_val_to_store);
    }
}

Value* call_add(CallASTNode* node, Function* F)
{
    MiniAPLArrayType type = TypeTable[node];
    int size = type.Cardinality();
    // Get an LLVM type for the flattened array.
    auto* vec_type = ArrayType::get(intTy(32), size);

    // Codegen arguments.
    Value* arg0 = node->Args[0]->codegen(F);
    Value* arg1 = node->Args[1]->codegen(F);

    auto alloc = Builder.CreateAlloca(vec_type);

    std::vector<int> curr_indices(type.dimension(), 0);
    int curr_array_idx = 0;
    map_over_array(type.dimensions, 0, curr_indices, alloc, curr_array_idx, [&arg0, &arg1](int idx, std::vector<Value*> gep_indices) {
        auto gep0 = Builder.CreateGEP(arg0, gep_indices);
        Value* elem0 = Builder.CreateLoad(gep0);

        auto gep1 = Builder.CreateGEP(arg1, gep_indices);
        Value* elem1 = Builder.CreateLoad(gep1);

        auto add = Builder.CreateAdd(elem0, elem1);
        return add;
    });

    return alloc;
}

Value* call_sub(CallASTNode* node, Function* F)
{
    MiniAPLArrayType type = TypeTable[node];
    int size = type.Cardinality();
    // Get an LLVM type for the flattened array.
    auto* vec_type = ArrayType::get(intTy(32), size);

    // Codegen arguments.
    Value* arg0 = node->Args[0]->codegen(F);
    Value* arg1 = node->Args[1]->codegen(F);

    auto alloc = Builder.CreateAlloca(vec_type);

    std::vector<int> curr_indices(type.dimension(), 0);
    int curr_array_idx = 0;
    map_over_array(type.dimensions, 0, curr_indices, alloc, curr_array_idx, [&arg0, &arg1](int idx, std::vector<Value*> gep_indices) {
        auto gep0 = Builder.CreateGEP(arg0, gep_indices);
        Value* elem0 = Builder.CreateLoad(gep0);

        auto gep1 = Builder.CreateGEP(arg1, gep_indices);
        Value* elem1 = Builder.CreateLoad(gep1);

        auto sub = Builder.CreateSub(elem0, elem1);
        return sub;
    });

    return alloc;
}

Value* call_neg(CallASTNode* node, Function* F)
{
    MiniAPLArrayType type = TypeTable[node];
    int size = type.Cardinality();
    // Get an LLVM type for the flattened array.
    auto* vec_type = ArrayType::get(intTy(32), size);

    Value* arg = node->Args[0]->codegen(F);

    auto* alloc = Builder.CreateAlloca(vec_type);

    std::vector<int> curr_indices(type.dimension(), 0);
    int curr_array_idx = 0;
    map_over_array(type.dimensions, 0, curr_indices, alloc, curr_array_idx, [&arg](int idx, std::vector<Value*> gep_indices) {
        auto gep = Builder.CreateGEP(arg, gep_indices);
        Value* elem = Builder.CreateLoad(gep);

        auto mul = Builder.CreateMul(elem, intConst(32, -1));
        return mul;
    });

    return alloc;
}

Value* call_exp(CallASTNode* node, Function* F)
{
    auto array = node->Args[0]->codegen(F);
    auto power = node->Args[1]->codegen(F);

    MiniAPLArrayType type = TypeTable[node];
    int size = type.Cardinality();

    auto* vec_type = ArrayType::get(intTy(32), size);
    auto* alloc = Builder.CreateAlloca(vec_type);

    std::vector<int> curr_indices(type.dimension(), 0);
    int curr_array_idx = 0;
    map_over_array(type.dimensions, 0, curr_indices, alloc, curr_array_idx, [&array, &power, &F](int idx, std::vector<Value*> gep_indices) {
        //load from array:
        auto gep = Builder.CreateGEP(array, gep_indices);
        Value* base = Builder.CreateLoad(gep);

        Value* result = intConst(32, 1); // initialize result to 1
        Value* counter = ConstantInt::get(intTy(32), 0); // initialize counter to 0
        BasicBlock* preheader = Builder.GetInsertBlock();
        BasicBlock* loop = BasicBlock::Create(TheContext, "exp-loop", F);
        BasicBlock* after = BasicBlock::Create(TheContext, "exp-loop-after", F);

        Builder.CreateBr(loop);
        Builder.SetInsertPoint(loop);

        PHINode* counter_phi = Builder.CreatePHI(intTy(32), 2);
        counter_phi->addIncoming(counter, preheader);
        PHINode* result_phi = Builder.CreatePHI(intTy(32), 2);
        result_phi->addIncoming(result, preheader);

        //result x base
        Value* mul = Builder.CreateMul(result_phi, base);
        result_phi->addIncoming(mul, loop);

        //increment counter:
        Value* inc_counter = Builder.CreateAdd(counter_phi, intConst(32, 1));
        counter_phi->addIncoming(inc_counter, loop);

        //exit loop if necessary
        Value* cmp_lt = Builder.CreateICmpSLT(counter_phi, power);
        Builder.CreateCondBr(cmp_lt, loop, after);
        Builder.SetInsertPoint(after);

        return result_phi;
    });

    return alloc;
}

Value* call_reduce(CallASTNode* node, Function* F)
{
    Value* arg = node->Args[0]->codegen(F);
    MiniAPLArrayType old_type = TypeTable[node->Args[0].get()];
    MiniAPLArrayType new_type = TypeTable[node];

    int old_size = old_type.Cardinality();
    int innermost_dimension_size = old_type.dimensions[old_type.dimensions.size() - 1];

    int new_size = old_size / innermost_dimension_size;

    auto* vec_type = ArrayType::get(intTy(32), new_size);
    auto* alloc = Builder.CreateAlloca(vec_type);

    std::vector<int> curr_indices(new_type.dimension(), 0);
    int curr_array_idx = 0;
    map_over_array(new_type.dimensions, 0, curr_indices, alloc, curr_array_idx, [&arg, innermost_dimension_size, &F](int idx, std::vector<Value*> _ignore) {
        Value* result = intConst(32, 0); // initialize result to 0
        Value* counter = ConstantInt::get(intTy(32), 0); // initialize counter to 0
        BasicBlock* preheader = Builder.GetInsertBlock();
        BasicBlock* loop = BasicBlock::Create(TheContext, "reduce-loop", F);
        BasicBlock* after = BasicBlock::Create(TheContext, "reduce-loop-after", F);

        Builder.CreateBr(loop);
        Builder.SetInsertPoint(loop);

        PHINode* counter_phi = Builder.CreatePHI(intTy(32), 2);
        counter_phi->addIncoming(counter, preheader);
        PHINode* result_phi = Builder.CreatePHI(intTy(32), 2);
        result_phi->addIncoming(result, preheader);

        cout << "index: " << idx << ", adding from original " << idx * innermost_dimension_size << " plus counter" << endl;
        Value* original_array_idx = Builder.CreateAdd(intConst(32, idx * innermost_dimension_size), counter_phi);
        std::vector<Value*> gep_indices { intConst(32, 0), original_array_idx };
        auto gep = Builder.CreateGEP(arg, gep_indices);
        Value* elem = Builder.CreateLoad(gep);

        //result + elem
        Value* add = Builder.CreateAdd(result_phi, elem);
        result_phi->addIncoming(add, loop);

        //increment counter:
        Value* inc_counter = Builder.CreateAdd(counter_phi, intConst(32, 1));
        counter_phi->addIncoming(inc_counter, loop);

        //exit loop if necessary
        Value* cmp_lt = Builder.CreateICmpSLT(counter_phi, intConst(32, innermost_dimension_size));
        Builder.CreateCondBr(cmp_lt, loop, after);
        Builder.SetInsertPoint(after);

        return result_phi;
    });

    return alloc;
}

Value* call_mkArray(CallASTNode* node, Function* F)
{
    std::unique_ptr<NumberASTNode> num_dimensions_ptr = std::unique_ptr<NumberASTNode>(static_cast<NumberASTNode*>(node->Args[0].release()));
    int num_dimensions = num_dimensions_ptr->Val;
    std::vector<int> dimensions;
    std::vector<Value*> array_values;
    for (int i = 1; i < num_dimensions + 1; i++) {
        std::unique_ptr<NumberASTNode> dimension_ptr = std::unique_ptr<NumberASTNode>(static_cast<NumberASTNode*>(node->Args[i].release()));
        dimensions.push_back(dimension_ptr->Val);
    }

    for (size_t i = num_dimensions + 1; i < node->Args.size(); i++) {
        Value* val = node->Args[i]->codegen(F);
        array_values.push_back(val);
    }

    int total_array_size = 1;
    for (int i = 0; i < num_dimensions; i++) {
        total_array_size *= dimensions[i];
    }

    auto* array_type = ArrayType::get(intTy(32), total_array_size);
    auto* alloc = Builder.CreateAlloca(array_type);

    std::vector<int> curr_indices(num_dimensions, 0);
    int curr_array_idx = 0;
    map_over_array(dimensions, 0, curr_indices, alloc, curr_array_idx, [&array_values](int idx, std::vector<Value*> _ignore) {
        return array_values[idx];
    });
    return alloc;
}

Value* CallASTNode::codegen(Function* F)
{
    // STUDENTS: FILL IN THIS FUNCTION
    if (Callee == "add") {
        return call_add(this, F);
    } else if (Callee == "sub") {
        return call_sub(this, F);
    } else if (Callee == "neg") {
        return call_neg(this, F);
    } else if (Callee == "exp") {
        return call_exp(this, F);
    } else if (Callee == "reduce") {
        return call_reduce(this, F);
    } else if (Callee == "mkArray") {
        return call_mkArray(this, F);
    } else {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Parser utilities
// ---------------------------------------------------------------------------
class ParseState {
public:
    int Position;
    vector<string> Tokens;

    ParseState(vector<string>& Tokens_)
        : Position(0)
        , Tokens(Tokens_)
    {
    }

    bool AtEnd()
    {
        return Position == (int)Tokens.size();
    }

    string peek()
    {
        if (AtEnd()) {
            return "";
        }
        return Tokens.at(Position);
    }

    string peek(const int Offset)
    {
        assert(Position + Offset < (int)Tokens.size());
        return Tokens.at(Position + Offset);
    }

    string eat()
    {
        auto Current = peek();
        Position++;
        return Current;
    }
};

std::ostream& operator<<(std::ostream& out, ParseState& PS)
{
    int i = 0;
    for (auto T : PS.Tokens) {
        if (i == PS.Position) {
            out << " | ";
        }
        out << T << " ";
        i++;
    }
    return out;
}

#define EAT(PS, t)                    \
    if (PS.eat() != (t)) {            \
        return LogError("EAT ERROR"); \
    }

unique_ptr<ASTNode> ParseExpr(ParseState& PS)
{
    string Name = PS.eat();
    if (is_int(Name)) {
        return unique_ptr<ASTNode>(new NumberASTNode(stoi(Name)));
    }

    if (PS.peek() == "(") {
        // Parse a function call

        PS.eat(); // consume "("

        vector<unique_ptr<ASTNode>> Args;
        while (PS.peek() != ")") {
            Args.push_back(ParseExpr(PS));
            if (PS.peek() != ")") {
                EAT(PS, ",");
            }
        }
        EAT(PS, ")");

        return unique_ptr<ASTNode>(new CallASTNode(Name, move(Args)));
    } else {
        return unique_ptr<ASTNode>(new VariableASTNode(Name));
    }
}

// ---------------------------------------------------------------------------
// Driver function for type-checking
// ---------------------------------------------------------------------------
void SetType(map<ASTNode*, MiniAPLArrayType>& Types, ASTNode* Expr)
{
    if (Expr->GetType() == EXPR_TYPE_FUNCALL) {
        CallASTNode* Call = static_cast<CallASTNode*>(Expr);
        for (auto& A : Call->Args) {
            SetType(Types, A.get());
        }

        if (Call->Callee == "mkArray") {
            int NDims = static_cast<NumberASTNode*>(Call->Args.at(0).get())->Val;
            vector<int> Dims;
            for (int i = 0; i < NDims; i++) {
                Dims.push_back(static_cast<NumberASTNode*>(Call->Args.at(i + 1).get())->Val);
            }
            Types[Expr] = { Dims };
        } else if (Call->Callee == "reduce") {
            Types[Expr] = Types[Call->Args.back().get()];
            Types[Expr].dimensions.pop_back();
        } else if (Call->Callee == "add" || Call->Callee == "sub") {
            Types[Expr] = Types[Call->Args.at(0).get()];
        } else {
            Types[Expr] = Types[Call->Args.at(0).get()];
        }
    } else if (Expr->GetType() == EXPR_TYPE_SCALAR) {
        Types[Expr] = { { 1 } };
    } else if (Expr->GetType() == EXPR_TYPE_VARIABLE) {
        string ExprName = static_cast<VariableASTNode*>(Expr)->Name;
        for (auto T : Types) {
            auto V = T.first;
            if (V->GetType() == EXPR_TYPE_VARIABLE) {
                string Name = static_cast<VariableASTNode*>(V)->Name;
                if (Name == ExprName) {
                    Types[Expr] = T.second;
                }
            }
        }
    }
}

int main(const int argc, const char** argv)
{
    assert(argc == 2);

    // Read in the source code file to a string
    string target_file = argv[1];

    std::ifstream t(target_file);
    std::string str((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    // Tokenize the file
    vector<string> Tokens;
    string NextToken = "";
    for (int i = 0; i < (int)str.size(); i++) {
        char NC = str[i];
        if (NC == ',' || NC == '(' || NC == ')' || NC == ';' || NC == '=') {
            if (NextToken != "") {
                Tokens.push_back(NextToken);
            }
            NextToken = string("") + NC;
            Tokens.push_back(NextToken);
            NextToken = "";
        } else if (!isspace(NC)) {
            NextToken += NC;
        } else {
            assert(isspace(NC));
            if (NextToken != "") {
                Tokens.push_back(NextToken);
            }
            NextToken = "";
        }
    }
    if (NextToken != "") {
        Tokens.push_back(NextToken);
    }

    vector<vector<string>> Stmts;
    vector<string> Toks;
    for (auto t : Tokens) {
        if (t == ";") {
            Stmts.push_back(Toks);
            Toks = {};
        } else {
            Toks.push_back(t);
        }
    }

    if (Toks.size() > 0) {
        Stmts.push_back(Toks);
    }

    // Parse each statement
    vector<unique_ptr<StmtAST>> ParsedStmts;
    for (auto S : Stmts) {
        ParseState PS(S);
        assert(S.size() > 0);
        if (PS.peek() != "assign") {
            unique_ptr<ASTNode> value = ParseExpr(PS);
            ParsedStmts.push_back(std::unique_ptr<StmtAST>(new ExprStmtAST(move(value))));
        } else {
            PS.eat(); // eat "assign"

            string Var = PS.eat();

            if (PS.eat() != "=") {
            } else {
                unique_ptr<ASTNode> value = ParseExpr(PS);
                ParsedStmts.push_back(std::unique_ptr<StmtAST>(new AssignStmtAST(Var, move(value))));
            }
        }
    }

    // Collect the statements into a program
    ProgramAST prog;
    prog.Stmts = move(ParsedStmts);

    // Infer types
    for (auto& S : prog.Stmts) {
        StmtAST* SA = S.get();
        if (SA->IsAssign()) {
            AssignStmtAST* Assign = static_cast<AssignStmtAST*>(SA);
            SetType(TypeTable, Assign->RHS.get());
            TypeTable[Assign->Name.get()] = TypeTable[Assign->RHS.get()];
        } else {
            ExprStmtAST* Expr = static_cast<ExprStmtAST*>(SA);
            SetType(TypeTable, Expr->Val.get());
        }
    }

    TheModule = llvm::make_unique<Module>("MiniAPL Module " + target_file, TheContext);
    std::vector<Type*> Args(0, Type::getDoubleTy(TheContext));
    FunctionType* FT = FunctionType::get(Type::getVoidTy(TheContext), Args, false);

    Function* F = Function::Create(FT, Function::ExternalLinkage, "__anon_expr", TheModule.get());
    BasicBlock::Create(TheContext, "entry", F);
    Builder.SetInsertPoint(&(F->getEntryBlock()));

    prog.Print(std::cout);
    prog.codegen(F);

    // NOTE: You may want to uncomment this line to see the LLVM IR you have generated
    // TheModule->print(errs(), nullptr);

    // Initialize the JIT, compile the module to a function,
    // find the function and then run it.
    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    InitializeNativeTargetAsmParser();
    TheJIT = llvm::make_unique<MiniAPLJIT>();
    InitializeModuleAndPassManager();
    auto H = TheJIT->addModule(std::move(TheModule));

    auto ExprSymbol = TheJIT->findSymbol("__anon_expr");
    void (*FP)() = (void (*)())(intptr_t)cantFail(ExprSymbol.getAddress());
    assert(FP != nullptr);
    FP();

    TheJIT->removeModule(H);

    return 0;
}
