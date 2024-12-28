//===- HECDialect.td - HEC Dialect Description ------------------*- tablegen -*-===//
//
// Copyright ...
// License ...
//
//===----------------------------------------------------------------------===//
//
//  This file defines the HEC dialect and its operations.
//
//===----------------------------------------------------------------------===//

#ifndef HEC_STRUCTURE_TD
#define HEC_STRUCTURE_TD

// -------------------------------------------------------------------------- //
// Standard MLIR Includes
// -------------------------------------------------------------------------- //
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Dialect/MLProgram/IR/MLProgramBase.td"
include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.td"
include "mlir/Dialect/MLProgram/IR/MLProgramTypes.td"

// -------------------------------------------------------------------------- //
// HEC Dialect Definition
// -------------------------------------------------------------------------- //

def HEC_Dialect : Dialect {
  let name = "hec";
  let cppNamespace = "mlir::hec";
  // Optionally, you can specify a custom summary/description for documentation.
  // let summary = "...";
  // let description = [{
  //   The HEC dialect ...
  // }];
}

// -------------------------------------------------------------------------- //
// Base Classes and Traits for HEC
// -------------------------------------------------------------------------- //


// A convenience class that enforces single-block, no-argument region containers.
class HECContainer<string mnemonic, list<Trait> traits = []>
    : HECOp<mnemonic,
            !listconcat(traits, [
               NoRegionArguments,
               NoTerminator,
               SingleBlock
            ])> {
  let regions = (region SizedRegion<1>: $body);
  let assemblyFormat = "attr-dict $body";

  let extraClassDeclaration = [{
    // Return the single block for a container that has exactly one region.
    ::mlir::Block *getBody() {
      return &getOperation()->getRegion(0).front();
    }
  }];
}

// -------------------------------------------------------------------------- //
// DesignOp
// -------------------------------------------------------------------------- //

def DesignOp : HECOp<"design", [
    NoRegionArguments,
    NoTerminator,
    SingleBlock,
    IsolatedFromAbove,
    SymbolTable
]> {
  let summary = "HEC Design";
  let description = [{
    The "hec.design" operation represents a top-level HEC design, containing
    zero or more HEC components. This must include a "main" component for an
    entry point.
  }];

  let arguments = (ins FlatSymbolRefAttr : $symbol);
  let regions = (region SizedRegion<1>: $body);

  let extraClassDeclaration = [{
    // Return the single block in the design.
    ::mlir::Block *getBodyBlock() {
      return &getOperation()->getRegion(0).front();
    }

    // Returns the "main" component, representing the entry point of the design.
    ComponentOp getMainComponent() {
      return lookupSymbol<ComponentOp>("main");
    }
  }];

  // Example: "hec.design @top ... "
  let assemblyFormat = "$symbol $body attr-dict";
  // let verifier = "return ::verify$cppClass(*this);";
}

// -------------------------------------------------------------------------- //
// ComponentOp
// -------------------------------------------------------------------------- //

def ComponentOp : HECOp<"component", [
    HasParent<"DesignOp">,
    Symbol,
    FunctionOpInterface,
    SingleBlock,
    NoTerminator
]> {
  let summary = "HEC Component";
  let description = [{
    The "hec.component" operation represents a wrapped/naked component in either
    STG or handshake (graph) style. It has a function-like signature describing
    its input and output ports, along with "interface" and "style" attributes.

    Examples:
    ```
    hec.component @simple_for { interface = "wrapped", style = "STG" } {
      ...
      hec.stateset {}
    }
    hec.component @irregular_for { interface = "wrapped", style = "handshake" } {
      ...
      hec.graph {}
    }
    ```
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    ArrayAttr        : $portNames,
    I64Attr          : $numInPorts,
    StrAttr          : $interfc,
    StrAttr          : $style
  );
  let results = (outs);

  let regions = (region SizedRegion<1> : $body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name,
               "ArrayRef<ComponentPortInfo>":$ports,
               "StringAttr":$interfc,
               "StringAttr":$style)>
  ];
 let extraClassDeclaration = [{
       //===------------------------------------------------------------------===//
       // FunctionOpInterface Methods
       //===------------------------------------------------------------------===//

       /// Returns the region on the current operation that is callable. This may
       /// return null in the case of an external callable object, e.g. an external
       /// function.
       ::mlir::Region *getCallableRegion() {
         return isExternal() ? nullptr : &getBody();
       }
       /// Returns the argument types of this function.
       ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
       /// Returns the result types of this function.
       ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
       //===------------------------------------------------------------------===//
       // RegionKindInterface Methods
       //===------------------------------------------------------------------===//
       static ::mlir::RegionKind getRegionKind(unsigned index) {
         return ::mlir::RegionKind::SSACFG;
       }
       //===------------------------------------------------------------------===//
       // SymbolOpInterface Methods
       //===------------------------------------------------------------------===//
       bool isDeclaration() { return isExternal(); }

           StateSetOp getStateSetOp();
             GraphOp getGraphOp();
 }];

  // Provide custom assembly format parsing/printing in C++.
  let hasCustomAssemblyFormat = 1;
  // let verifier = "return ::verify$cppClass(*this);";
}

// -------------------------------------------------------------------------- //
// InstanceOp
// -------------------------------------------------------------------------- //

def InstanceOp : HECOp<"instance", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    HasParent<"ComponentOp">
]> {
  let summary = "HEC Instance";
  let description = [{
    The "hec.instance" operation instantiates a child HEC component within the
    current component, referencing it by symbol name. Example:

    ```
    %in, %out = hec.instance "myInst" @MyComponent : i64, i16
    ```
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    FlatSymbolRefAttr:$componentName
  );
  let results = (outs Variadic<AnyType>:$results);

  // Example format:   hec.instance "myInst" @MyComponent : i64, i64
  let assemblyFormat = "$instanceName $componentName attr-dict (`:` type($results)^)?";
  // let verifier = "return ::verify$cppClass(*this);";

  let extraClassDeclaration = [{
    /// Lookup the component for the symbol. Returns null on invalid name.
    ComponentOp getReferencedComponent();
  }];

}

// -------------------------------------------------------------------------- //
// PrimitiveOp
// -------------------------------------------------------------------------- //

def PrimitiveOp : HECOp<"primitive", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    ParentOneOf<["ComponentOp", "DesignOp"]>
]> {
  let summary = "HEC Primitive Instance";
  let description = [{
    The "hec.primitive" operation represents an instance of a built-in HEC
    primitive operator (e.g., a register, an adder, etc.). Example:

    ```
    %reg_in, %reg_out = hec.primitive "myReg" is "register" : i32, i32
    ```
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    StrAttr:$primitiveName
  );
  let results = (outs Variadic<AnyType>:$results);

  // Example format:   hec.primitive "myReg" is "register" : i32, i32
  let assemblyFormat = "$instanceName `is` $primitiveName attr-dict (`:` type($results)^)?";
  // let verifier = "return ::verify$cppClass(*this);";

  let extraClassDeclaration = [{
    // Return the port information for the named primitive (e.g. "register").
    SmallVector<ComponentPortInfo> getPrimitivePortInfo();
  }];
}

// -------------------------------------------------------------------------- //
// AssignOp
// -------------------------------------------------------------------------- //

def AssignOp : HECOp<"assign", [
    ParentOneOf<["StateSetOp", "StateOp", "GraphOp", "StageOp"]>
]> {
  let summary = "HEC Assign";
  let description = [{
    The "hec.assign" operation represents a non-blocking assignment, which may
    optionally be guarded by a boolean condition.
  }];

  let arguments = (ins
    AnyType : $dest,
    AnyType : $src,
    Optional<I1> : $guard
  );

  // Format example:   %dest = %src if %guard : i32 -> i32
  let assemblyFormat = "$dest `=` $src (`if` $guard^)? attr-dict `:` type($src) `->` type($dest)";
}

// -------------------------------------------------------------------------- //
// WireOp
// -------------------------------------------------------------------------- //

def WireOp : HECOp<"wire", []> {
  let summary = "HEC Wire";
  let description = [{
    The `hec.wire` operation declares a new wire variable.
  }];

  let arguments = (ins
    StrAttr : $name
  );
  let results = (outs
    AnyType : $out
  );

  // Example format:   hec.wire "fooWire" : i64
  let assemblyFormat = "$name attr-dict `:` type($out)";
}

// -------------------------------------------------------------------------- //
// EnableOp
// -------------------------------------------------------------------------- //

def EnableOp : HECOp<"enable", []> {
  let summary = "HEC Enable";
  let description = [{
    The "hec.enable" operation drives a port (or wire) with the value 1.
    Optionally guarded by a condition.
  }];

  let arguments = (ins
    AnyType : $port,
    Optional<I1> : $cond
  );
  // Example format:   hec.enable %port if %cond : i1
  let assemblyFormat = "$port (`if` $cond^)? attr-dict `:` type($port)";
}

// -------------------------------------------------------------------------- //
// GoOp
// -------------------------------------------------------------------------- //

def GoOp : HECOp<"go", []> {
  let summary = "HEC Go";
  let description = [{
    The "hec.go" operation represents the beginning of an instance's execution.
  }];

  let arguments = (ins
    StrAttr : $name,
    Optional<I1> : $cond
  );

  // Example format:  hec.go "myInstance" if %cond
  let assemblyFormat = "$name (`if` $cond^)? attr-dict";
}

// -------------------------------------------------------------------------- //
// End of File
// -------------------------------------------------------------------------- //

#endif // HEC_STRUCTURE_TD
