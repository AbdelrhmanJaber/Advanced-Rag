import multiprocessing as mp
from pathlib import Path
import logging
from typing import Any, List, Tuple
from multiprocessing.synchronize import Lock as LockType
from dataclasses import dataclass
import time
import argparse
from src.ingestion.loader import LocalFileLoader
from src.ingestion.validator import validator
from src.preprocessing.classifier import classifier
from src.preprocessing.native_parser import NativeDocumentParser
from src.preprocessing.scanned_parser import ScannedDocumentParser

SENTINEL = None


@dataclass
class pipliningConfig :
    raw_dir : str
    output_dir : str
    n_validator_workers : int = 4
    n_classifer_workers : int = 4
    n_native_parser_workers : int = 4
    n_scanned_parser_workers : int = 2
    queue_max_size : int = 200


# lock and release shared queues to prevent concurrency issues
# args : counter , locker , downstreams (queue/s and number of workders of next stage)
def lock_and_release(counter : Any , lock : LockType , downstream : List[Tuple[mp.Queue,int]])->None:
    with lock :
        counter.value -= 1
        is_last = counter.value == 0
        if is_last :
            for q , n_consumers in downstream :
                for _ in range(n_consumers):
                    q.put(SENTINEL)
    
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orchestrator")
 

# loader
# args : raw dirctory , queue out (of validator) , number of workers of validator to send signals

def loader_stage(raw_dir : str , q_out : mp.Queue , num_of_validator_workers : int) -> None :
    loader = LocalFileLoader(raw_dir)
    count = 0
    try : 
        for file_path in loader.scan_files() :
            q_out.put(str(file_path))
            count +=1
    finally :
        logger.info("loader found  %d file pathes",count)
        for _ in range(num_of_validator_workers): #None for each worker of the next stage
            q_out.put(SENTINEL)


#validation stage 
#args : worker_id ,  q_in from loader stage (file pathes) , q_out of classifier , counter , locker , downstream
def validator_stage(worker_id : int , q_in : mp.Queue , q_out : mp.Queue ,
                     counter : Any , locker : LockType , downstream : List[Tuple[mp.Queue,int]]) -> None :
    n_ok , n_rejected , n_errors = 0,0,0
    while True :
        item = q_in.get()
        if item == SENTINEL:
            break
        try : 
            is_valid , metadata , message = validator(item)
            if is_valid:
                q_out.put(item)
                n_ok+=1
            else :
                logger.info("[validate-%d] rejected %s -> %s", worker_id, item, message)
                n_rejected += 1
        except Exception:
            logger.exception("[validate-%d] error on %s", worker_id, item)
            n_errors += 1
    logger.info("[validate-%d] done ok=%d rejected=%d errors=%d", worker_id, n_ok, n_rejected, n_errors)
    lock_and_release(counter , locker , downstream)


# classifier stage
# args : worker_id , q_in (from validatot contains file pathes) , native_queue , scanned_queue , counter , locker , downstream
def classifier_stage(worker_id : int , q_in : mp.Queue , q_native_out : mp.Queue , q_scanned_out : mp.Queue ,
                     counter : Any , locker : LockType , downstream : List[Tuple[mp.Queue,int]]) -> None:
    n_native , n_scanned , n_errors = 0,0,0
    while True:
        item = q_in.get()
        if item == SENTINEL:
            break
        try :
            metadata = classifier(item)
            if metadata["file_type"] == "NATIVE":
                q_native_out.put(item)
                n_native += 1
            else :
                q_scanned_out.put(item)
                n_scanned +=1
        except Exception:
                    logger.exception("[classifier-%d] error on %s", worker_id, item)
                    n_errors += 1   
    logger.info("[classifier-%d] done native=%d scanned=%d errors=%d", worker_id, n_native, n_scanned, n_errors)
    lock_and_release(counter , locker , downstream)


# native parse stage
# args : worker_id , q_in from the output of classifier stage that contain file pathes , output_dirctory
def native_parse_stage(worker_id : int , q_in : mp.Queue , output_dir : str) ->  None :
    n_ok , n_errors = 0 , 0
    while True :
        item = q_in.get()
        if item == SENTINEL:
            break
        try:
            file_output_dir = Path(output_dir) / "native" / Path(item).stem
            parser = NativeDocumentParser(item, output_dir=str(file_output_dir))
            out_path = parser.parse_and_save_json()
            logger.info("[native-%d] saved %s", worker_id, out_path)
            n_ok += 1
        except Exception:
            logger.exception("[native-%d] error on %s", worker_id, item)
            n_errors += 1
    logger.info("[native-%d] done ok=%d errors=%d", worker_id, n_ok, n_errors)

# scanned parse stage
# args : worker_id , q_in from classifier that contain file pathes , output directory

def scanned_parse_stage(worker_id : int , q_in : mp.Queue , output_dir : str) -> None :
    n_ok , n_errors = 0 , 0
    while True :
        item = q_in.get()
        if item == SENTINEL :
            break
        try:
            file_output_dir = Path(output_dir) / "scanned" / Path(item).stem
            parser = ScannedDocumentParser(item , output_dir=str(file_output_dir))
            out_path = parser.parse_and_save_json()
            logger.info("[scanned-%d] saved %s", worker_id, out_path)
            n_ok += 1
        except Exception:
            logger.exception("[scanned-%d] error on %s", worker_id, item)
            n_errors += 1
    logger.info("[scanned-%d] done ok=%d errors=%d", worker_id, n_ok, n_errors)



# Orchestrator of piplining

def run_pipline(config: pipliningConfig)  -> None :
    #define the queues of the pipline
    q_validator: mp.Queue = mp.Queue(config.queue_max_size)
    q_classifier: mp.Queue = mp.Queue(config.queue_max_size)
    q_native_parser: mp.Queue = mp.Queue(config.queue_max_size)
    q_scanned_parser: mp.Queue = mp.Queue(config.queue_max_size)

    #define the counter and lockers
    validator_counter = mp.Value("i" , config.n_validator_workers)
    validator_lock = mp.Lock()

    classifier_counter = mp.Value("i" , config.n_classifer_workers)
    classifier_lock = mp.Lock()

    #define processes list
    processes : List[mp.Process] = []

    # stage 1
    processes.append(mp.Process(target=loader_stage , args=(config.raw_dir ,q_validator , config.n_validator_workers),name="loader"))

    #stage 2
    for i in range(config.n_validator_workers):
        processes.append(mp.Process(
                target=validator_stage ,
                args=(
                    i,q_validator , q_classifier , validator_counter , validator_lock , [(q_classifier , config.n_classifer_workers)] 
                ),
                name="validator"
            ))
    #stage 3 : take the verified pathes from stage 2
    for i in range(config.n_classifer_workers):
        processes.append(mp.Process(
            target=classifier_stage,
            args=(
                i,q_classifier , q_native_parser , q_scanned_parser , classifier_counter , classifier_lock , 
                [(q_native_parser , config.n_native_parser_workers) , (q_scanned_parser , config.n_scanned_parser_workers)]
            ),
            name="classifier"
        ))

    # stage 4 parse native docs and save it as json using tree builder
    for i in range(config.n_native_parser_workers):
        processes.append(mp.Process(
            target=native_parse_stage,
            args=(i , q_native_parser , config.output_dir),
            name="native-parser"
        ))

    # stage 5 parse scanned docs and save it as json using tree builder
    for i in range(config.n_scanned_parser_workers):
        processes.append(mp.Process(
            target=scanned_parse_stage,
            args=(i,q_scanned_parser,config.output_dir),
            name="scanned-parser"
        ))

    start = time.time()
    logger.info(
        "Starting pipeline | validate=%d classify=%d native=%d scanned=%d",
        config.n_validator_workers, config.n_classifer_workers,
        config.n_native_parser_workers, config.n_scanned_parser_workers,
    )
 
    for p in processes:
        p.start()
    for p in processes :
        p.join()

    logger.info("Pipeline finished in %.2fs", time.time() - start)


def parse_args() -> pipliningConfig:
    parser = argparse.ArgumentParser(description="Parallel ingestion + preprocessing pipeline")
    parser.add_argument("--raw-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--n_validator_workers", type=int, default=4)
    parser.add_argument("--n_classifer_workers", type=int, default=4)
    parser.add_argument("--n_native_parser_workers", type=int, default=4)
    parser.add_argument("--n_scanned_parser_workers", type=int, default=2)
    parser.add_argument("--queue_max_size", type=int, default=200)
    args = parser.parse_args()
 
    return pipliningConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        n_validator_workers = args.n_validator_workers ,
        n_classifer_workers = args.n_classifer_workers,
        n_native_parser_workers = args.n_native_parser_workers,
        n_scanned_parser_workers = args.n_scanned_parser_workers,
        queue_max_size = args.queue_max_size
    )
 
 
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    cfg = parse_args()
    run_pipline(cfg)
