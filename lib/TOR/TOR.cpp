#include "TOR/TOR.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"    // for function_interface_impl
//#include "mlir/Dialect/Func/IR/FunctionImplementation.h"

#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "TOR/TOR.cpp.inc"
#include "TOR/TOR.h.inc"

using namespace mlir;
using namespace tor;

static LogicalResult verifyDesignOp(tor::DesignOp op)
{
  // TODO: check the existance of a `main` func
  return success();
}

//LogicalResult tor::DesignOp::verify()
//{
//    // TODO: check the existance of a `main` func
//    return success();
//}

LogicalResult tor::CallOp::verify()
{
    // TODO: check the existance of a `main` func
    return success();
}
// Lookup a symbol referenced by MyCallOp
LogicalResult tor::CallOp::verifySymbolUses(::mlir::SymbolTableCollection &symbolTable) {
    // Assuming 'op' is an instance of MyCallOp
    FlatSymbolRefAttr callee = getCalleeAttr();

    // Resolve the symbol reference to its definition
    auto sym = symbolTable.lookupNearestSymbolFrom(getOperation(), callee);
    if (!sym) {
        return emitOpError("callee '") << callee.getRootReference()
                                       << "' not found in symbol table";
    }

    // Optionally, verify that the resolved symbol is of the expected type
    // For example, ensure it's a function operation
    if (!sym->hasTrait<::mlir::detail::FunctionOpInterfaceTrait>()) {
        return emitOpError("callee '") << callee.getRootReference()
                                       << "' is not a function";
    }

    return success();


}

void tor::AddIOp::build(OpBuilder &odsBuilder,
                         OperationState &odsState,
                         Value lhs,
                         Value rhs) 
{
  IntegerType lhsType = lhs.getType().cast<IntegerType>();
  IntegerType rhsType = rhs.getType().cast<IntegerType>();
  IntegerType resType = IntegerType::get(odsState.getContext(), std::max(lhsType.getWidth(), rhsType.getWidth()));
  odsState.addAttribute("starttime", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  odsState.addAttribute("endtime", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  return build(odsBuilder, odsState, resType, ValueRange{lhs, rhs}, ArrayRef<NamedAttribute>{});
}

void tor::SubIOp::build(OpBuilder &odsBuilder,
                         OperationState &odsState,
                         Value lhs,
                         Value rhs) 
{
  IntegerType lhsType = lhs.getType().cast<IntegerType>();
  IntegerType rhsType = rhs.getType().cast<IntegerType>();
  IntegerType resType = IntegerType::get(odsState.getContext(), std::max(lhsType.getWidth(), rhsType.getWidth()));
  odsState.addAttribute("starttime", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  odsState.addAttribute("endtime", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  return build(odsBuilder, odsState, resType, ValueRange{lhs, rhs}, ArrayRef<NamedAttribute>{});
}

void tor::MulIOp::build(OpBuilder &odsBuilder,
                         OperationState &odsState,
                         Value lhs,
                         Value rhs) 
{
  IntegerType lhsType = lhs.getType().cast<IntegerType>();
  IntegerType rhsType = rhs.getType().cast<IntegerType>();
  IntegerType resType = IntegerType::get(odsState.getContext(), std::min(64U, lhsType.getWidth() + rhsType.getWidth()));

  odsState.addAttribute("starttime", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  odsState.addAttribute("endtime", odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  return build(odsBuilder, odsState, resType, ValueRange{lhs, rhs}, ArrayRef<NamedAttribute>{});
}

void tor::FuncOp::build(::mlir::OpBuilder &odsBuilder,
                         ::mlir::OperationState &odsState,
                         StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<NamedAttrList> argAttrs)
{
  odsState.addAttribute(SymbolTable::getSymbolAttrName(),
                        odsBuilder.getStringAttr(name));
  odsState.addAttribute("function_type", TypeAttr::get(type));
  odsState.attributes.append(attrs.begin(), attrs.end());
  odsState.addRegion();
}

 void FuncOp::print(::mlir::OpAsmPrinter &p)
{
  FunctionType funcType = getFunctionType();

    // Define the names of the attributes

    auto op = this->getOperation();
    StringRef typeAttrName = "function_type";
    StringAttr argAttrsName = op->getAttrOfType<StringAttr>("arg_attrs");
    StringAttr resAttrsName = op->getAttrOfType<StringAttr>("res_attrs");

    function_interface_impl::printFunctionSignature(
            p,                  // OpAsmPrinter &
            *this,                 // Operation *
            funcType.getInputs(),    // ArrayRef<Type> for argument types
            /*isVariadic=*/true,     // bool
            funcType.getResults()    // ArrayRef<Type> for result types
            // Optionally: argument attrs, result attrs, custom printing callbacks...
    );

    p.printRegion(this->getRegion());


  //mlir::function_interface_impl::printFunctionOp(p, *this, true, typeAttrName, argAttrsName,resAttrsName);
 p.printOptionalAttrDict(this->getOperation()->getAttrs(), {"name"});

}

::mlir::ParseResult FuncOp::parse(::mlir::OpAsmParser &parser,
                                  ::mlir::OperationState &result) {
    // This lambda builds a FunctionType from a list of arg/result Types.
    auto buildFuncType = [](Builder &builder,
                            ArrayRef<Type> argTypes,
                            ArrayRef<Type> resTypes,
                            function_interface_impl::VariadicFlag /*unused*/,
                            std::string & /*unused*/) -> FunctionType {
        return builder.getFunctionType(argTypes, resTypes);
    };

    // Provide the attribute names parseFunctionOp needs:
    // - The name of the function-type attribute (e.g. "function_type")
    // - The attribute names for arg- and result-attrs (if you want to support them)
    auto &b = parser.getBuilder();
    StringAttr typeAttrName = b.getStringAttr("function_type");
    StringAttr argAttrsName = b.getStringAttr("arg_attrs");
    StringAttr resAttrsName = b.getStringAttr("res_attrs");

    // Now call parseFunctionOp with all needed arguments.
    if (failed(function_interface_impl::parseFunctionOp(
            parser,
            result,
            /*allowVariadic=*/true,
            /*typeAttrName=*/typeAttrName,
            /*funcTypeBuilder=*/buildFuncType,
            /*argAttrsName=*/argAttrsName,
            /*resAttrsName=*/resAttrsName))) {
        return failure();
    }

    // Optionally parse trailing attributes (if your dialect supports them).
    if (parser.parseOptionalAttrDict(result.attributes))
        return failure();

    return success();
}

::mlir::LogicalResult FuncOp::verify()
{
  auto fnInputTypes = getFunctionType().getInputs();
  Block &entryBlock = front();

  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
  {
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "module signature(" << fnInputTypes[i] << ')';
  }
  return success();
}

void ReturnOp::print(OpAsmPrinter &p)
{
  p << "tor.return";
  if (getNumOperands() != 0)
  {
    p << ' ';
    p.printOperands(getOperands());
  }
  /*
  p << " at " << time();*/
  if (getNumOperands() != 0)
  {
    p << " : ";
    interleaveComma(getOperandTypes(), p);
  }
}

ParseResult ReturnOp::parse(OpAsmParser &parser, OperationState &result)
{
  SmallVector<OpAsmParser::UnresolvedOperand, 1> opInfo;
  SmallVector<Type, 1> types;
  ::mlir::IntegerAttr timeAttr;
  llvm::SMLoc loc = parser.getCurrentLocation();
  return failure(parser.parseOperandList(opInfo) ||
                 /* parser.parseKeyword("at") ||
                 parser.parseAttribute(timeAttr, 
                    parser.getBuilder().getIntegerType(32), 
                    "time", result.attributes) || */
                 (!opInfo.empty() && parser.parseColonTypeList(types)) ||
                 parser.resolveOperands(opInfo, types, loc, result.operands));
}

LogicalResult ReturnOp::verify()
{
  auto parent = getParentOp();

  StringRef parentName = parent->getName().getStringRef();

  if (parentName.equals(StringRef("tor.func")))
  {
    auto function = dyn_cast<tor::FuncOp>(&parent);
    // if (!function)
    //   return emitOpError("must have a handshake.func parent");

    // The operand number and types must match the function signature.
    const auto &results = function->getResultTypes();
    if (getNumOperands() != results.size())
      return emitOpError("has ")
             << getNumOperands()
             << " operands, but enclosing function returns " << results.size();

    for (unsigned i = 0, e = results.size(); i != e; ++i)
      if (getOperand(i).getType() != results[i])
        return emitError()
               << "type of return operand " << i << " ("
               << getOperand(i).getType()
               << ") doesn't match function result type (" << results[i] << ")";

    return success();
  }
  return emitOpError("must have a tor.func or tor.module parent");
}

 void TimeGraphOp::print(OpAsmPrinter &p)
{

  p << TimeGraphOp::getOperationName() << " (" << getStarttime() << " to " << getEndtime() << ")";

  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult TimeGraphOp::parse(OpAsmParser &parser, OperationState &result)
{
  result.regions.reserve(1);
  Region *region = result.addRegion();

  ::mlir::IntegerAttr starttime;
  ::mlir::IntegerAttr endtime;

  OpAsmParser::UnresolvedOperand cond;

  if (/*parser.parseKeyword("on") || */ parser.parseLParen() ||
      parser.parseAttribute(starttime,
                            parser.getBuilder().getIntegerType(32),
                            "starttime", result.attributes) ||
      parser.parseKeyword("to") ||
      parser.parseAttribute(endtime,
                            parser.getBuilder().getIntegerType(32),
                            "endtime", result.attributes) ||
      parser.parseRParen())
  {
    return failure();
  }

  if (/*parser.parseKeyword("then") ||*/
      parser.parseRegion(*region, {}, {}))
  {
    return failure();
  }

  TimeGraphOp::ensureTerminator(*region, parser.getBuilder(), result.location);
  return success();
}

//LogicalResult StartTimeOp::verify()
//{
//  return success();
//}
//
// LogicalResult EndTimeOp::verify()
//{
//  return success();
//}
//
//LogicalResult SuccTimeOp::verify()
//{
//  return success();
//}

void IfOp::print(OpAsmPrinter &p)
{
  bool printBlockTerminators = false;

  p << IfOp::getOperationName() << " " << getCondition()
    << " on (" << getStarttime() << " to " << getEndtime() << ")";

  if (!getResults().empty())
  {
    p << " -> (" << getResultTypes() << ")";
    printBlockTerminators = true;
  }

  p << " then";

  p.printRegion(getThenRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  auto &elseRegion = getElseRegion();

  if (!elseRegion.empty())
  {
    p << " else ";
    p.printRegion(elseRegion,
                  /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/printBlockTerminators);
  }

  p.printOptionalAttrDict(getOperation()->getAttrs(), {"starttime", "endtime"});
}

ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result)
{
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  ::mlir::IntegerAttr starttime;
  ::mlir::IntegerAttr endtime;

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;

  Type i1Type = builder.getIntegerType(1);
  if (parser.parseOperand(cond) ||
      parser.resolveOperand(cond, i1Type, result.operands))
  {
    return failure();
  }

  if (parser.parseKeyword("on") || parser.parseLParen() ||
      parser.parseAttribute(starttime,
                            parser.getBuilder().getIntegerType(32),
                            "starttime", result.attributes) ||
      parser.parseKeyword("to") ||
      parser.parseAttribute(endtime,
                            parser.getBuilder().getIntegerType(32),
                            "endtime", result.attributes) ||
      parser.parseRParen())
  {
    return failure();
  }
  if (parser.parseOptionalArrowTypeList(result.types))
  {
    return failure();
  }

  if (parser.parseKeyword("then") ||
      parser.parseRegion(*thenRegion, {}, {}))
  {
    return failure();
  }
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  if (!parser.parseOptionalKeyword("else"))
  {
    if (parser.parseRegion(*elseRegion, {}, {}))
    {
      return failure();
    }
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}
/*
 LogicalResult IfOp::verify()
{
  return success();
}
*/
void tor::ForOp::build(OpBuilder &builder, OperationState &result, Value lb,
                        Value ub, Value step,
                        IntegerAttr starttime, IntegerAttr endtime,
                        ValueRange iterArgs,
                        BodyBuilderFn bodyBuilder)
{
  result.addAttribute("starttime", starttime);
  result.addAttribute("endtime", endtime);
  result.addOperands({lb, ub, step});
  result.addOperands(iterArgs);
  for (Value v : iterArgs)
    result.addTypes(v.getType());
  result.addRegion();
}

/// Prints the initialization list in the form of
///   <prefix>(%inner = %outer, %inner2 = %outer2, <...>)
/// where 'inner' values are assumed to be region arguments and 'outer' values
/// are regular SSA values.
static void printInitializationList(OpAsmPrinter &p,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "")
{
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it)
                        { p << std::get<0>(it) << " = " << std::get<1>(it); });
  p << ")";
}

void ForOp::print(OpAsmPrinter &p)
{
  p << getOperationName() << " " << getInductionVar() << " = "
    << "(" << getLowerBound() << " : " << getLowerBound().getType() << ")"
    << " to " 
    << "(" << getUpperBound() << " : " << getUpperBound().getType() << ")" 
    << " step " << "(" << getStep() << " : " << getStep().getType() << ")";

  p.printNewline();
  p << "on (" << getStarttime() << " to " << getEndtime() << ")";

  printInitializationList(p, getRegionIterArgs(), getIterOperands(), " iter_args");

  if (!getIterOperands().empty())
    p << " -> (" << getIterOperands().getTypes() << ")";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/hasIterOperands());
  p.printOptionalAttrDict(this->getOperation()->getAttrs(), {"starttime", "endtime"});
}

 ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result)
{
  auto &builder = parser.getBuilder();
  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;
  ::mlir::IntegerAttr starttime, endtime;

  // Parse the induction variable followed by '='.
  if (parser.parseArgument(inductionVariable) || parser.parseEqual())
    return failure();

  // Parse loop bounds.
  //Type indexType = builder.getIndexType();
  Type lbType, ubType, stepType;
  if (parser.parseLParen() || parser.parseOperand(lb) ||
      parser.parseColon() || parser.parseType(lbType) || parser.parseRParen() ||
      parser.resolveOperand(lb, lbType, result.operands))
      return failure();

  if (parser.parseKeyword("to"))
    return failure();
  
  if (parser.parseLParen() || parser.parseOperand(ub) ||
      parser.parseColon() || parser.parseType(ubType) || parser.parseRParen() ||
      parser.resolveOperand(ub, ubType, result.operands))
      return failure();

  if (parser.parseKeyword("step"))
    return failure();

  if (parser.parseLParen() || parser.parseOperand(step) ||
      parser.parseColon() || parser.parseType(stepType) || parser.parseRParen() ||
      parser.resolveOperand(step, stepType, result.operands))
      return failure();
  /*
  Type indexType = builder.getIndexType();
  if (parser.parseOperand(lb) ||
      parser.resolveOperand(lb, indexType, result.operands) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) ||
      parser.resolveOperand(ub, indexType, result.operands) ||
      parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(step, indexType, result.operands))
    return failure();
  */
  // Parse "on [starttime, endtime]"
  if (parser.parseKeyword("on") || parser.parseLParen() ||
      parser.parseAttribute(starttime,
                            parser.getBuilder().getIntegerType(32),
                            "starttime", result.attributes) ||
      parser.parseKeyword("to") ||
      parser.parseAttribute(endtime,
                            parser.getBuilder().getIntegerType(32),
                            "endtime", result.attributes) ||
      parser.parseRParen())
    return failure();

  // Parse the optional initial iteration arguments.
    SmallVector<OpAsmParser::UnresolvedOperand, 4>  operands;
    SmallVector<OpAsmParser::Argument, 4> regionArgs;

      SmallVector<Type, 4> argTypes;
  regionArgs.push_back(inductionVariable);

  if (succeeded(parser.parseOptionalKeyword("iter_args")))
  {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    // Resolve input operands.
    for (auto operand_type : llvm::zip(operands, result.types))
      if (parser.resolveOperand(std::get<0>(operand_type),
                                std::get<1>(operand_type), result.operands))
        return failure();
  }

  // Induction variable.
  Type iterType = builder.getIntegerType(std::max(lbType.getIntOrFloatBitWidth(), 
      std::max(ubType.getIntOrFloatBitWidth(), stepType.getIntOrFloatBitWidth())));

  argTypes.push_back(iterType);
  // Loop carried variables
  argTypes.append(result.types.begin(), result.types.end());
  // Parse the body region.
  Region *body = result.addRegion();
  if (regionArgs.size() != argTypes.size())
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");

  if (parser.parseRegion(*body, regionArgs, false))
    return failure();

  tor::ForOp::ensureTerminator(*body, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}
static LogicalResult verifyForOp(tor::ForOp op)
{
  return success();
}

void WhileOp::print(OpAsmPrinter &p)
{

  p << getOperationName();
  printInitializationList(p, getBefore().front().getArguments(), getInits(),
                          " ");
  p.printNewline();
  p << "on (" << getStarttime() << " to " << getEndtime() << ")";
  p << " : ";
  p.printFunctionalType(getInits().getTypes(), getResults().getTypes());
  p.printRegion(getBefore(), /*printEntryBlockArgs=*/false);
  p << " do";
  p.printRegion(getAfter());
    p.printOptionalAttrDict(getOperation()->getAttrs(), {"starttime", "endtime"});
}

ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result)
{
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    SmallVector<OpAsmParser::Argument, 4> regionArgs;
  Region *before = result.addRegion();
  Region *after = result.addRegion();

  OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, operands);
  if (listResult.has_value() && failed(listResult.value()))
    return failure();

  ::mlir::IntegerAttr starttime, endtime;
  if (parser.parseKeyword("on") || parser.parseLParen() ||
      parser.parseAttribute(starttime,
                            parser.getBuilder().getIntegerType(32),
                            "starttime", result.attributes) ||
      parser.parseKeyword("to") ||
      parser.parseAttribute(endtime,
                            parser.getBuilder().getIntegerType(32),
                            "endtime", result.attributes) ||
      parser.parseRParen())
  {
    return failure();
  }

  FunctionType functionType;
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseColonType(functionType)))
    return failure();

  result.addTypes(functionType.getResults());

  if (functionType.getNumInputs() != operands.size())
  {
    return parser.emitError(typeLoc)
           << "expected as many input types as operands "
           << "(expected " << operands.size() << " got "
           << functionType.getNumInputs() << ")";
  }

  // Resolve input operands.
  if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  return failure(
      parser.parseRegion(*before, regionArgs, false) ||
      parser.parseKeyword("do") || parser.parseRegion(*after) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes));
}

/// Verifies that two ranges of types match, i.e. have the same number of
/// entries and that types are pairwise equals. Reports errors on the given
/// operation in case of mismatch.
template <typename OpTy>
static LogicalResult verifyTypeRangesMatch(OpTy op, TypeRange left,
                                           TypeRange right, size_t lbias,
                                           size_t rbias, StringRef message)
{
  if (left.size() + lbias != right.size() + rbias)
    return op.emitError("expects the same number of ") << message;

  for (unsigned i = 0, e = left.size(); i + lbias < e; ++i)
  {
    if (left[i + lbias] != right[i + rbias])
    {
      InFlightDiagnostic diag = op.emitError("expects the same types for ")
                                << message;
      diag.attachNote() << "for argument " << i << ", found " << left[i + lbias]
                        << " and " << right[i + rbias];
      return diag;
    }
  }

  return success();
}

/// Verifies that the first block of the given `region` is terminated by a
/// `TerminatorTy`. Reports errors on the given operation if it is not the case.
template <typename TerminatorTy>
static TerminatorTy verifyAndGetTerminator(tor::WhileOp op, Region &region,
                                           StringRef errorMessage) {
    Operation *terminatorOperation = region.front().getTerminator();
    if (auto yield = dyn_cast_or_null<TerminatorTy>(terminatorOperation))
        return yield;

    // Emit an error on the `op` itself, rather than calling a free function.
    auto diag = op.emitOpError() << errorMessage;
    if (terminatorOperation)
        diag.attachNote(terminatorOperation->getLoc())
                << "terminator here";

    // Return nullptr for a failed cast (assuming TerminatorTy is a pointer-like type).
    return nullptr;
}

LogicalResult WhileOp::verify() {
  // if (failed(RegionBranchOpInterface::verifyTypes(op)))
  //   return failure();

  auto beforeTerminator = verifyAndGetTerminator<tor::ConditionOp>(
      *this, getBefore(),
      "expects the 'before' region to terminate with 'tor.condition'");
  if (!beforeTerminator)
    return failure();

  TypeRange trailingTerminatorOperands = beforeTerminator.getArgs().getTypes();
  if (failed(verifyTypeRangesMatch(*this, trailingTerminatorOperands,
                                   getAfter().getArgumentTypes(), 0, 0,
                                   "trailing operands of the 'before' block "
                                   "terminator and 'after' region arguments")))
    return failure();

  if (failed(verifyTypeRangesMatch(
          *this, trailingTerminatorOperands, getResultTypes(), 0, 0,
          "trailing operands of the 'before' block terminator and op results")))
    return failure();

  auto afterTerminator = verifyAndGetTerminator<tor::YieldOp>(
      *this, getAfter(),
      "expects the 'after' region to terminate with 'tor.yield'");
  return success(afterTerminator != nullptr);
}

#define GET_OP_CLASSES
#include "TOR/TOR.cpp.inc"
