# Applied-Mathematical-Theories-and-Methods

## Part 2 summary (6 runs)

This project runs six mandatory experiments for Part 2: three on the synthetic regression task (Problem R) and three on MNIST01 (Problem M). The runs are:

- **Problem R (synthetic regression)**
  1. GD_ARMIJO (max_epochs=250)
  2. SGD (max_epochs=60, batch_size=128, lr=0.1, step schedule)
  3. KFAC (max_epochs=35, batch_size=256, lr=0.2, damping=1e-2)
- **Problem M (MNIST01 classification)**
  4. GD_ARMIJO (max_epochs=60)
  5. SGD (max_epochs=15, batch_size=128, lr=0.1)
  6. KFAC (max_epochs=10, batch_size=256, lr=0.2, damping=1e-2)

## Dependencies

- numpy
- pandas
- torch
- torchvision (optional but recommended; required to download/load MNIST)

## How to run

```bash
python run_all_part2.py --seed 0 --dtype float64 --device cpu
```

## Output CSV schema and mapping to report tables

The script writes two CSVs:

- `results_part2_synthetic.csv`
- `results_part2_mnist01.csv`

Each CSV uses a strict schema with the following columns:

```
run_name,dataset,optimizer,activation,m,d,seed,dtype,device,N_train,N_test,l2_reg,max_epochs,batch_size,lr,schedule,step_gamma,step_every,damping,update_freq,eta0,beta,c,grad_tol,patience,min_delta,epochs_run,time_s,final_train_loss,final_test_loss,train_acc,test_acc,final_grad_norm,status,error_msg
```

Mapping to report tables:

- **Table 10.1**: synthetic regression runs (Problem R), pulled from `results_part2_synthetic.csv`.
- **Table 10.2**: MNIST01 classification runs (Problem M), pulled from `results_part2_mnist01.csv`.
- **Table 10.3**: optimizer diagnostics (time, gradient norms, and status/error fields) drawn from the same CSVs, using columns such as `time_s`, `final_grad_norm`, `status`, and `error_msg`.

## MNIST download failure behavior

If MNIST cannot be downloaded or torchvision is unavailable, the MNIST rows are still written with `status=NO_DATA` and an `error_msg` describing the failure.

## Packaging instruction

Zip all `.py` files and this README into a single archive named `name-ID.zip` (replace with your actual name and ID).
