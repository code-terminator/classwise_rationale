aspect=0
balance="True"
output_dir="./beer_results/aspect"$aspect

data_dir="./data/beer_review/beer"$aspect
annotation_path="./data/beer_review/annotations.json"
embedding_dir="./embeddings"
embedding_name="glove.6B.100d.txt"

batch_size=200
visual_interval=10000

sparsity_percentage=0.08
sparsity_lambda=5.
continuity_lambda=10.
num_epchos=30

python run_beer.py --data_dir $data_dir --balance $balance --aspect $aspect --output_dir $output_dir --embedding_dir $embedding_dir --embedding_name $embedding_name --annotation_path $annotation_path  --batch_size $batch_size  --sparsity_percentage $sparsity_percentage --num_epchos $num_epchos --sparsity_lambda $sparsity_lambda  --continuity_lambda $continuity_lambda --visual_interval $visual_interval