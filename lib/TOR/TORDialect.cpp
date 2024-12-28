#include "TOR/TORDialect.h"
#include "TOR/TOR.h"
#include "TOR/TORTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "TOR/TORDialect.cpp.inc"


using namespace mlir;
using namespace mlir::tor;

struct TORInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final {
    return true;
  }

  void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const {
    auto returnOp = cast<tor::ReturnOp>(op);

    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (auto it : llvm::enumerate(returnOp.getOperands())) {
        auto index = it.index();
        auto value  = valuesToRepl[index];
        value.replaceAllUsesWith(it.value());
    }
  }
};

void TORDialect::initialize()
{
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "TOR/TOR.cpp.inc"
      >();
  addInterfaces<TORInlinerInterface>();
}

// Provide implementations for the enums we use.
#include "TOR/TOREnums.cpp.inc"

