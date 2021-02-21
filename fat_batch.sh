for epsilon in 0 1 2 4 8 16 32
do
python run_fat.py --epsilon_adv epsilon --n_global_rounds 5
python run_at.py --epsilon_adv epsilon --n_global_rounds 40
done