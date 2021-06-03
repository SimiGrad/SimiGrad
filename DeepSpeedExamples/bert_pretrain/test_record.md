lamb_seq128_adaptive_from128_target0.75_halfbatch_smooth0.9_lr0.0004_test1:
use uneven step size, which is increase by 0.5 but decrease by 1

lamb_seq128_adaptive_from128_target0.75_halfbatch_lr0.0004_test2:
        lr_this_step = config["training"]["learning_rate"] * decay_linear(global_step=current_data_sample_count,total_steps=config["training"]["total_training_steps"]*65536)
                if(self.cos_placeholder<similarity_target):
                    self.change_gradient_accumulation_steps(max(1,self.gradient_accumulation_steps()+1,int(self.gradient_accumulation_steps()*1.1)),smooth=False)
                elif(self.cos_placeholder>similarity_target and self.gradient_accumulation_steps()>1):
                    self.change_gradient_accumulation_steps(max(1,self.gradient_accumulation_steps()-1,int(self.gradient_accumulation_steps()*0.9)),smooth=False)


JOB_NAME=lamb_seq128_adaptive_from128_target0.75_halfbatch_lr0.0004_test3
        --lr_offset 0 \


JOB_NAME=lamb_seq128_adaptive_from128_target0.75_halfbatch_lr0.0006
            "learning_rate": 0.0006,


JOB_NAME=lamb_seq128_adaptive_from128_target0.75_halfbatch_lr0.0006_test2
        lr_this_step = config["training"]["learning_rate"] * decay_exp(global_step=current_data_sample_count, decay_rate=config["training"]["decay_rate"],decay_steps=config["training"]["decay_step"]*65536*1.5)


old:
        "128": {
            "num_epochs": 150,
            "warmup_proportion": 0.06,
            "learning_rate": 0.0006,
            "num_workers": 0,
            "async_worker": true,
            "decay_rate": 0.90,
            "decay_step": 250,
            "total_training_steps": 7500
        },

JOB_NAME=lamb_seq128_adaptive_from128_target0.75_halfbatch_lr0.0006_test2
        "128": {
            "num_epochs": 150,
            "warmup_proportion": 0.06,
            "learning_rate": 11e-3,
            "num_workers": 0,
            "async_worker": true,
            "decay_rate": 0.90,
            "decay_step": 250,
            "total_training_steps": 7500
        },
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_exp(
                global_step_for_lr/65536, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])