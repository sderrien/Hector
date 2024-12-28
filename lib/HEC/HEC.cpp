#include "HEC/HEC.h"

#include <iostream>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h" // Modern MLIR header for function-like ops
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace hec;
using namespace mlir::function_interface_impl;

//===----------------------------------------------------------------------===//
// DesignOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyDesignOp(DesignOp design) {
    // if (!design.getMainComponent())
    //   return design.emitOpError("must contain one \"main\" component");
    return success();
}

//===----------------------------------------------------------------------===//
// ComponentOp
//===----------------------------------------------------------------------===//

// Return the type of a given component as a function type.
static FunctionType getComponentType(ComponentOp component) {
    return component.getFunctionType();
}

// Return the port information for a given component.
SmallVector<ComponentPortInfo> mlir::hec::getComponentPortInfo(Operation *op) {
    assert(isa<ComponentOp>(op) && "Can only get port info from a ComponentOp");
    auto component = cast<ComponentOp>(op);
    auto portTypes = getComponentType(component).getInputs();
    auto portNamesAttr = component.getPortNames();
    uint64_t numInPorts = component.getNumInPorts();

    SmallVector<ComponentPortInfo> results;
    results.reserve(portNamesAttr.size());
    for (uint64_t i = 0, e = portNamesAttr.size(); i < e; ++i) {
        auto dir = (i < numInPorts) ? PortDirection::INPUT : PortDirection::OUTPUT;
        results.push_back({portNamesAttr[i].cast<StringAttr>(), portTypes[i], dir});
    }
    return results;
}

// Implement the custom parser for ComponentOp.
ParseResult ComponentOp::parse(OpAsmParser &parser, OperationState &result) {
    // Parse symbol name for this component.
    StringAttr componentName;
    if (parser.parseSymbolName(componentName,
                               SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    // Parse the signature, i.e. ports: ( inputPorts... ) -> ( outputPorts... )
    SmallVector<OpAsmParser::Argument> ports;
    SmallVector<Type> portTypes;

    // Local lambda to parse a parenthesized list of ports.
    auto parsePortDefList = [&](bool &failedParsing) -> ParseResult {
        if (parser.parseLParen())
            return failure();
        SmallVector<OpAsmParser::Argument> localPorts;
        SmallVector<Type> localTypes;
        if (succeeded(parser.parseOptionalRParen())) {
            // no ports at all: "()"
            return success();
        }

        do {
            OpAsmParser::Argument port;
            Type portType;
            if (failed(parser.parseArgument(port)) ||
                failed(parser.parseColon()) ||
                failed(parser.parseType(portType))) {
                failedParsing = true;
                return failure();
            }
            localPorts.push_back(port);
            localTypes.push_back(portType);
        } while (succeeded(parser.parseOptionalComma()));

        if (parser.parseRParen())
            return failure();

        // Append to the global lists.
        ports.append(localPorts.begin(), localPorts.end());
        portTypes.append(localTypes.begin(), localTypes.end());
        return success();
    };

    bool parseFailed = false;
    if (parsePortDefList(parseFailed))
        return failure();
    size_t numInPorts = ports.size(); // record # input ports

    // Parse the arrow -> for the outputs.
    if (parser.parseArrow())
        return failure();
    if (parsePortDefList(parseFailed))
        return failure();
    if (parseFailed)
        return failure();

    // Build the function-like type for the component.
    auto type = parser.getBuilder().getFunctionType(portTypes, {});
    result.addAttribute("function_type", TypeAttr::get(type));

    // Build portNames from the SSA name (without '%').
    auto *context = parser.getBuilder().getContext();
    SmallVector<Attribute> portNames;
    portNames.reserve(ports.size());
    for (auto &port : ports) {
        StringRef name = port.ssaName.name;
        if (name.startswith("%"))
            name = name.drop_front();
        portNames.push_back(StringAttr::get(context, name));
    }
    result.addAttribute("portNames", ArrayAttr::get(context, portNames));
    result.addAttribute("numInPorts",
                        parser.getBuilder().getI64IntegerAttr(numInPorts));

    // Now parse the style: { interface="xxx", style="yyy" }
    if (parser.parseLBrace() || parser.parseKeyword("interface") ||
        parser.parseEqual())
        return failure();
    StringAttr interfc, style;
    if (parser.parseAttribute(interfc))
        return failure();
    if (parser.parseComma() || parser.parseKeyword("style") ||
        parser.parseEqual())
        return failure();
    if (parser.parseAttribute(style))
        return failure();
    if (parser.parseRBrace())
        return failure();
    result.addAttribute("interfc", interfc);
    result.addAttribute("style", style);

    // Parse the body region.
    Region *body = result.addRegion();
    if (parser.parseRegion(*body, ports))
        return failure();
    if (body->empty())
        body->emplaceBlock();

    // Parse optional extra attributes.
    NamedAttrList extraAttrs;
    if (failed(parser.parseOptionalAttrDict(extraAttrs)))
        return failure();
    for (auto &attr : extraAttrs)
        result.addAttribute(attr.getName(), attr.getValue());

    // Set the symbol name.
    result.addAttribute(SymbolTable::getSymbolAttrName(), componentName);
    return success();
}

// Implement the custom printer for ComponentOp.
void ComponentOp::print(OpAsmPrinter &p) {
    // Print the op name + symbol name.
    p << getOperationName() << ' ';
    p.printSymbolName(getName());

    // Print the signature: ( inputPorts ) -> ( outputPorts )
    // We know how many input ports vs output ports from `numInPorts`.
    // Gather the in/out from getArguments().
    uint64_t nIn = getNumInPorts();
    uint64_t nTotal = getNumArguments();
    auto args = getArguments();
    p << " (";
    if (nTotal == 0) {
        p << ") -> ()";
    } else if (nIn == 0) {
        p << ") -> (";
    }

    for (unsigned i = 0; i < nTotal; ++i) {
        p.printOperand(args[i]);
        p << " : " << args[i].getType();
        if (i + 1 == nIn)
            p << ") -> (";
        else if (i + 1 < nTotal)
            p << ", ";
        else if (i + 1 == nTotal)
            p << ")";
    }

    // Print the attributes inside braces: { interface="...", style="..." }
    p << " {interface=\"" << getInterfc() << "\", style=\"" << getStyle() << "\"} ";

    // Print the body region.
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);

    // Print any other attributes, but skip the ones we've already handled.
    SmallVector<StringRef> elided = {
            "sym_name", "portNames", "numInPorts", "type",
            "interfc", "style"
    };
    p.printOptionalAttrDict(getOperation()->getAttrs(), elided);
}

static LogicalResult verifyComponentOp(ComponentOp op) {
    // Example check: verify the # of in-ports matches numInPorts attribute.
    SmallVector<ComponentPortInfo> ports = getComponentPortInfo(op);
    uint64_t expected = op->getAttrOfType<IntegerAttr>("numInPorts").getInt();
    uint64_t actual = llvm::count_if(ports, [](auto &info){
        return info.direction == PortDirection::INPUT;
    });
    if (expected != actual)
        return op.emitOpError("mismatched number of in ports")
                << " (expected " << expected << ", got " << actual << ")";

    return success();
}

StateSetOp ComponentOp::getStateSetOp() {
    return *getBody().getOps<StateSetOp>().begin();
}

GraphOp ComponentOp::getGraphOp() {
    return *getBody().getOps<GraphOp>().begin();
}

void ComponentOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ComponentPortInfo> ports,
                        StringAttr interfc, StringAttr style) {
    // Symbol name:
    result.addAttribute(SymbolTable::getSymbolAttrName(), name);

    // Collect port info
    SmallVector<Type, 8> portTypes;
    SmallVector<Attribute, 8> portNames;
    uint64_t numInPorts = 0;

    for (auto &port : ports) {
        if (port.direction == PortDirection::INPUT)
            ++numInPorts;
        portNames.push_back(port.name);
        portTypes.push_back(port.type);
    }

    // Build a function type for the component.
    auto fnType = builder.getFunctionType(portTypes, {});
    result.addAttribute("function_type", TypeAttr::get(fnType));

    // Record port names and number of inputs.
    result.addAttribute("portNames", builder.getArrayAttr(portNames));
    result.addAttribute("numInPorts", builder.getI64IntegerAttr(numInPorts));
    result.addAttribute("interfc", interfc);
    result.addAttribute("style", style);

    // Create a region with one block.
    Region *body = result.addRegion();
    Block &block = body->emplaceBlock();
    block.addArguments(portTypes, SmallVector<Location>(portTypes.size(), result.location));

    // Insert stateful ops as needed:
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);
    if (style.getValue() == "STG") {
        auto ss = builder.create<StateSetOp>(result.location);
        ss.getRegion().emplaceBlock();
    } else if (style.getValue() == "pipeline") {
        auto sset = builder.create<StageSetOp>(result.location);
        sset.getRegion().emplaceBlock();
    } else {
        builder.create<GraphOp>(result.location);
    }
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

ComponentOp InstanceOp::getReferencedComponent() {
    auto design = (*this)->getParentOfType<DesignOp>();
    if (!design)
        return nullptr;
    return design.lookupSymbol<ComponentOp>(getComponentName());
}
/*
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    auto comp = getReferencedComponent();
    if (!comp)
        return;
    auto portNames = comp.getPortNames();
    std::string prefix = getInstanceName().str() + ".";
    for (size_t i = 0, e = portNames.size(); i < e; ++i) {
        StringRef portName = portNames[i].cast<StringAttr>().getValue();
        setNameFn(getResult(i), prefix + portName.str());
    }
}
 LogicalResult InstanceOp::verify() {
    auto instance = this;
    if (instance.getComponentName() == "main")
        return instance.emitOpError("cannot reference the main component.");

    // Ensure the referenced component exists.
    auto comp = instance.getReferencedComponent();
    if (!comp)
        return instance.emitOpError("references non-existent component '")
                << instance.getComponentName() << "'";

    // Ensure no recursive instantiation.
    auto parent = instance->getOperation()->PParentOfType<ComponentOp>();
    if (parent == comp)
        return instance.emitOpError("recursive instantiation of parent component");

    // Verify result ports match the referenced component type.
    SmallVector<ComponentPortInfo> portInfos = getComponentPortInfo(comp);
    if (instance.getNumResults() != portInfos.size())
        return instance.emitOpError("wrong # of results");

    for (size_t i = 0, e = portInfos.size(); i < e; ++i) {
        if (instance.getResult(i).getType() != portInfos[i].type)
            return instance.emitOpError("result type mismatch on port ")
                    << portInfos[i].name;
    }
    return success();
}
*/

//===----------------------------------------------------------------------===//
// PrimitiveOp
//===----------------------------------------------------------------------===//

SmallVector<ComponentPortInfo> PrimitiveOp::getPrimitivePortInfo() {
    // ... Implementation of your large if/else chain ...
    // Returns a SmallVector<ComponentPortInfo> for the requested primitive name.
    // e.g. "register", "add_integer", "cmp_float", etc.

    SmallVector<ComponentPortInfo> results;
    StringAttr name = getPrimitiveNameAttr();

    if (name.getValue() == "register") {
        results.push_back({
                                  StringAttr::get(getContext(), "reg"),
                                  IntegerType::get(getContext(), 32),
                                  PortDirection::INOUT
                          });
    } else if (name.getValue() == "add_integer" ||
               name.getValue() == "sub_integer" ||
               name.getValue() == "mul_integer" ||
               name.getValue() == "div_integer") {
        results.push_back({StringAttr::get(getContext(), "operand0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "operand1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "trunc_integer") {
        results.push_back({StringAttr::get(getContext(), "operand"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("cmp_integer")) {
        results.push_back({StringAttr::get(getContext(), "operand0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "operand1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           IntegerType::get(getContext(), 1),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "add_float" ||
               name.getValue() == "sub_float" ||
               name.getValue() == "mul_float" ||
               name.getValue() == "div_float") {
        results.push_back({StringAttr::get(getContext(), "operand0"),
                           FloatType::getF32(getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "operand1"),
                           FloatType::getF32(getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           FloatType::getF32(getContext()),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("cmp_float")) {
        results.push_back({StringAttr::get(getContext(), "operand0"),
                           FloatType::getF32(getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "operand1"),
                           FloatType::getF32(getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           IntegerType::get(getContext(), 1),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "sitofp") {
        results.push_back({StringAttr::get(getContext(), "operand"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           FloatType::getF32(getContext()),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "fptosi") {
        results.push_back({StringAttr::get(getContext(), "operand"),
                           FloatType::getF32(getContext()),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("mem")) {
        auto rw = (*this)->getAttr("ports").cast<StringAttr>();
        assert(rw != nullptr && "Must provide read/write for mem");
        if (rw.getValue() == "r") {
            results.push_back({StringAttr::get(getContext(), "r_en"),
                               getType(0), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "addr"),
                               getType(1), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "r_data"),
                               getType(2), PortDirection::OUTPUT});
        } else if (rw.getValue() == "w") {
            results.push_back({StringAttr::get(getContext(), "w_en"),
                               getType(0), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "addr"),
                               getType(2), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "w_data"),
                               getType(3), PortDirection::INPUT});
        } else if (rw.getValue() == "rw") {
            results.push_back({StringAttr::get(getContext(), "w_en"),
                               getType(0), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "r_en"),
                               getType(1), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "addr"),
                               getType(2), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "w_data"),
                               getType(3), PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "r_data"),
                               getType(4), PortDirection::OUTPUT});
        }
    } else if (name.getValue() == "buffer") {
        results.push_back({StringAttr::get(getContext(), "dataIn"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "merge") {
        results.push_back({StringAttr::get(getContext(), "dataIn.0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataIn.1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "branch") {
        results.push_back({StringAttr::get(getContext(), "condition"),
                           IntegerType::get(getContext(), 1),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataIn"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut.0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut.1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue().contains("load")) {
        results.push_back({StringAttr::get(getContext(), "address_in"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "data_out"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get(getContext(), "address_out"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get(getContext(), "data_in"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "control"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
    } else if (name.getValue().contains("store")) {
        results.push_back({StringAttr::get(getContext(), "address_in"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "data_in"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "address_out"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get(getContext(), "data_out"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get(getContext(), "control"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
    } else if (name.getValue().contains("fork")) {
        if (name.getValue().contains(":")) {
            results.push_back({StringAttr::get(getContext(), "dataIn"),
                               IntegerType::get(getContext(), 32),
                               PortDirection::INPUT});
            std::string forkName = name.getValue().str();
            unsigned num =
                    std::atoi(forkName.substr(forkName.find(":") + 1).c_str());
            for (unsigned idx = 0; idx < num; ++idx) {
                results.push_back({StringAttr::get(getContext(), "dataOut." + std::to_string(idx)),
                                   IntegerType::get(getContext(), 32),
                                   PortDirection::OUTPUT});
            }
        } else {
            results.push_back({StringAttr::get(getContext(), "dataIn"),
                               IntegerType::get(getContext(), 32),
                               PortDirection::INPUT});
            results.push_back({StringAttr::get(getContext(), "dataOut.0"),
                               IntegerType::get(getContext(), 32),
                               PortDirection::OUTPUT});
            results.push_back({StringAttr::get(getContext(), "dataOut.1"),
                               IntegerType::get(getContext(), 32),
                               PortDirection::OUTPUT});
        }
    } else if (name.getValue().contains("fifo")) {
        results.push_back({StringAttr::get(getContext(), "dataIn"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
    } else if (name.getValue().contains("dyn_Mem")) {
        if (name.getValue().contains(":")) {
            std::string memName = name.getValue().str();
            unsigned loadnum =
                    std::atoi(memName
                                      .substr(memName.find(":") + 1,
                                              memName.find(",") - memName.find(":") - 1)
                                      .c_str());
            unsigned storenum =
                    std::atoi(memName.substr(memName.find(",") + 1).c_str());
            for (unsigned idx = 0; idx < loadnum; ++idx) {
                results.push_back({StringAttr::get(getContext(), "load_address." + std::to_string(idx)),
                                   IntegerType::get(getContext(), 32),
                                   PortDirection::INPUT});
                results.push_back({StringAttr::get(getContext(), "load_data." + std::to_string(idx)),
                                   IntegerType::get(getContext(), 32),
                                   PortDirection::OUTPUT});
            }
            for (unsigned idx = 0; idx < storenum; ++idx) {
                results.push_back({StringAttr::get(getContext(), "store_address." + std::to_string(idx)),
                                   IntegerType::get(getContext(), 32),
                                   PortDirection::INPUT});
                results.push_back({StringAttr::get(getContext(), "store_data." + std::to_string(idx)),
                                   IntegerType::get(getContext(), 32),
                                   PortDirection::INPUT});
            }
        }
    } else if (name.getValue() == "control_merge") {
        results.push_back({StringAttr::get(getContext(), "dataIn.0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataIn.1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get(getContext(), "condition"),
                           IntegerType::get(getContext(), 1),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "shift_left") {
        results.push_back({StringAttr::get(getContext(), "operand0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "operand1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "select") {
        results.push_back({StringAttr::get(getContext(), "condition"),
                           IntegerType::get(getContext(), 1),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataIn.0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataIn.1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "constant") {
        results.push_back({StringAttr::get(getContext(), "control"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataIn"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "mux_dynamic") {
        results.push_back({StringAttr::get(getContext(), "dataIn.0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataIn.1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
        results.push_back({StringAttr::get(getContext(), "condition"),
                           IntegerType::get(getContext(), 1),
                           PortDirection::INPUT});
    } else if (name.getValue() == "and") {
        results.push_back({StringAttr::get(getContext(), "operand0"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "operand1"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "result"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "fptosi") {
        results.push_back({StringAttr::get(getContext(), "dataIn"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "neg_float") {
        results.push_back({StringAttr::get(getContext(), "dataIn"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
        results.push_back({StringAttr::get(getContext(), "dataOut"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::OUTPUT});
    } else if (name.getValue() == "sink") {
        results.push_back({StringAttr::get(getContext(), "dataIn"),
                           IntegerType::get(getContext(), 32),
                           PortDirection::INPUT});
    } else {
        std::cerr << name.getValue().str() << std::endl;
        assert(0 && "hec.primitive op has an undefined primitiveName");
    }

    return results;
}
/*
void PrimitiveOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
    auto portInfos = getPrimitivePortInfo();
    std::string prefix = getInstanceName().str() + ".";
    assert(portInfos.size() == getResults().size() &&
           "# of results must match the primitive's definition");
    for (size_t i = 0, e = portInfos.size(); i < e; ++i) {
        setNameFn(getResult(i), prefix + portInfos[i].name.getValue().str());
    }
}
*/
static LogicalResult verifyPrimitiveOp(PrimitiveOp op) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// StateSetOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyStateSetOp(StateSetOp op) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

ParseResult StateOp::parse(OpAsmParser &parser, OperationState &result) {
    // Parse the state symbol name.
    StringAttr stateName;
    if (parser.parseSymbolName(stateName,
                               SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    // The old snippet used parseOptionalStar in a reversed meaning:
    // If '*' is present => initial=0, else => initial=1.
    IntegerAttr initial;
    if (succeeded(parser.parseOptionalStar())) {
        initial = parser.getBuilder().getIntegerAttr(
                parser.getBuilder().getI1Type(), 0);
    } else {
        initial = parser.getBuilder().getIntegerAttr(
                parser.getBuilder().getI1Type(), 1);
    }
    result.addAttribute("initial", initial);

    // Parse the body region.
    Region *body = result.addRegion();
    if (parser.parseRegion(*body))
        return failure();
    if (body->empty())
        body->emplaceBlock();

    return success();
}

void StateOp::print(OpAsmPrinter &p) {
    // Print "hec.state symbolName"
    p << getOperationName() << " ";
    p.printSymbolName(getName());

    // If 'initial' is 0 => print '*'
    if (getInitial())
        p << "*";

    // Print the body region.
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);
}

static LogicalResult verifyStateOp(StateOp op) {
    // ...
    return success();
}

void StateOp::build(OpBuilder &builder, OperationState &result,
                    StringAttr name, IntegerAttr initial) {
    result.addAttribute(SymbolTable::getSymbolAttrName(), name);
    result.addAttribute("initial", initial);

    Region *body = result.addRegion();
    body->emplaceBlock();
}

//===----------------------------------------------------------------------===//
// TransitionOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyTransitionOp(TransitionOp transition) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// GotoOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyGotoOp(GotoOp gotoop) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// CDoneOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyCDoneOp(CDoneOp done) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// DoneOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyDoneOp(DoneOp done) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyGraphOp(GraphOp graph) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// StageSetOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyStageSetOp(StageSetOp stageset) {
    // ...
    return success();
}

//===----------------------------------------------------------------------===//
// StageOp
//===----------------------------------------------------------------------===//

ParseResult StageOp::parse(OpAsmParser &parser, OperationState &result) {
    // Parse symbol name for the stage.
    StringAttr stageName;
    if (parser.parseSymbolName(stageName,
                               SymbolTable::getSymbolAttrName(),
                               result.attributes))
        return failure();

    // Parse the body region.
    Region *body = result.addRegion();
    if (parser.parseRegion(*body))
        return failure();
    if (body->empty())
        body->emplaceBlock();

    return success();
}

void StageOp::print(OpAsmPrinter &p) {
    p << getOperationName() << " ";
    p.printSymbolName(getName());
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
            /*printBlockTerminators=*/true,
            /*printEmptyBlock=*/false);
}

static LogicalResult verifyStageOp(StageOp stage) {
    // ...
    return success();
}

void StageOp::build(OpBuilder &builder, OperationState &result,
                    StringAttr name) {
    result.addAttribute(SymbolTable::getSymbolAttrName(), name);
    result.addRegion()->emplaceBlock();
}

//===----------------------------------------------------------------------===//
// Include generated op classes
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "HEC/HEC.cpp.inc"
