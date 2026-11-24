// What a model measured about itself on this machine, kept between loads: its recorded routing, which
// the planner prices the cache on (bmoe/routing.h, AutoPlanOptions::routing), and the verdicts of the
// decisions --auto measures (bmoe/measured_knobs.h). Every front-end keeps them the same way, so an
// --auto plan is the same whichever one runs it, and what one of them measured is the others' too.
// Where the folder is is the front-end's decision (bmoe-cli and bmoe-server default to the per-user data
// folder); inside it, one file per model named by the model file's content fingerprint
// (model_fingerprint in bmoe/probe.h): a file copied or renamed finds its record, another model never
// does. The fingerprint is a few MiB of
// reads, so it is computed once per file and remembered by path, size and modification time, the key
// the model library already uses for what it reads from a file.
#pragma once

#include "bmoe/measured_knobs.h"
#include "bmoe/routing.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace bmoe {

class ModelRecords {
public:
    // `dir` empty: nothing is kept or read.
    explicit ModelRecords(std::string dir) : dir_(std::move(dir)) {}

    // The record kept for the model at `model_path`, if there is one that reads.
    bool load(const std::string & model_path, RoutingTrace & out);

    // Keep `rec` for that model, unless the one already kept decoded further: a short session does
    // not replace a longer record. Written beside and renamed, so a reader never sees half of one.
    void keep(const std::string & model_path, const RoutingTrace & rec);

    // The file a model's record lives in; "" when the model cannot be fingerprinted.
    std::string path_for(const std::string & model_path, const char * ext = ".routing");

    // The knob trial's verdicts kept for the model, one per machine and engine (bmoe/measured_knobs.h),
    // beside its routing. `keep_knobs` replaces the record of the machine and engine `rec` names.
    bool load_knobs(const std::string & model_path, std::vector<MeasuredKnobs> & out);
    void keep_knobs(const std::string & model_path, const MeasuredKnobs & rec);

private:
    void write(const std::string & path, const std::string & text, const char * what);
    struct Key {
        uint64_t size = 0;
        long long mtime = 0;
        std::string fingerprint;
    };
    std::string dir_;
    std::mutex m_;
    // A verdict is read, merged and written as one: more than one session may keep verdicts at once.
    std::mutex knobs_m_;
    std::map<std::string, Key> fingerprints_;
};

} // namespace bmoe
