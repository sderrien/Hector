#ifndef HEC_DIALECT_TD
#define HEC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Dialect/MLProgram/IR/MLProgramBase.td"
include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.td"
include "mlir/Dialect/MLProgram/IR/MLProgramTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

 include "mlir/IR/OpBase.td"
 include "mlir/IR/BuiltinTypes.td"
 include "mlir/IR/SymbolInterfaces.td"
 include "mlir/Interfaces/CallInterfaces.td"

def HECDialect : Dialect {
  let name = "hec";
  let summary = "Types and operations for hec dialect";
  let description = [{
    This dialect defines the 'hec' dialect, which ...
  }];
  let cppNamespace = "::mlir::hec";
  
  let extraClassDeclaration = [{
    /// Register all hec types.
    void registerTypes();
  }];
}

class HECOp<string mnemonic,list<Trait> traits = []> :

    Op<HECDialect, mnemonic, traits>;



#endif // HEC_DIALECT_TD