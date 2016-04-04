#define HAVE_ATLAS
#define HAVE_MEMALIGN
#define HAVE_POSIX_MEMALIGN

#include <algorithm>
#include <hmm/transition-model.h>
#include <iostream>
#include <memory>
#include <tree/context-dep.h>
#include <util/common-utils.h>

class TimitModel {
 public:
  TimitModel()
    : trans_model_(new kaldi::TransitionModel),
      ctx_dep_(new kaldi::ContextDependency) {
    LoadTransitionModel();
    LoadContextDependencyModel();

    // We have 48 phones in TIMIT.
    const std::vector<int32>& phones = trans_model_->GetTopo().GetPhones();
    assert(phones.size() == 48);

    std::vector<int32> num_pdf_classes(1 + *std::max_element(phones.begin(), phones.end()), -1);
    for (size_t i = 0; i < phones.size(); ++i) {
      num_pdf_classes[phones[i]] = trans_model_->GetTopo().NumPdfClasses(phones[i]);
    }

    std::vector<std::vector<std::pair<int32, int32>>> pdf_info;
    ctx_dep_->GetPdfInfo(phones, num_pdf_classes, &pdf_info);
    for (size_t pdfid = 0; pdfid  < pdf_info.size(); ++pdfid) {
      const std::vector<std::pair<int32, int32>>& info = pdf_info[pdfid];
      for (size_t c = 0; c < info.size(); ++c) {
        const int32 phone = info[c].first;
        const int32 pdf_class = info[c].second;
        // std::cout << "pdfid: " << pdfid << ", " << phone << ", " << pdf_class << std::endl;
        // std::cout << pdfid << " " << phone << std::endl;
        assert(phone <= phones.size());
      }
    }
  }

  void LoadTransitionModel() {
    kaldi::ReadKaldiObject(
        "/data-local/wchan/kaldi/egs/timit/s5/exp/sgmm2_4_ali/final.mdl", trans_model_.get());

    for (int32 trans_id = 1; trans_id <= trans_model_->NumTransitionIds(); ++trans_id) {
      int32 phone_id = trans_model_->TransitionIdToPhone(trans_id);
      std::cout << trans_id << " " << phone_id << std::endl;
    }
  }

  void LoadContextDependencyModel() {
    kaldi::ReadKaldiObject(
        "/data-local/wchan/kaldi/egs/timit/s5/exp/sgmm2_4_ali/tree", ctx_dep_.get());
  }

 private:
  std::unique_ptr<kaldi::TransitionModel> trans_model_;
  std::unique_ptr<kaldi::ContextDependency> ctx_dep_;
};

int main(int argc, char* argv[]) {
  std::unique_ptr<TimitModel> timit(new TimitModel);
}
