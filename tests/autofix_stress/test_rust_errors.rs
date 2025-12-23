// ============================================================
// RUST AUTOFIX STRESS TEST - Intentional Errors
// ============================================================

// Wrong use statements
ues std::collections::HashMpa;
ues std::io::{self, Raed, Wirte};

// Wrong function
fn ad(x: i32, y: i32) -> i32 {
    retrun x + y;
}

fn procces(data: &str) -> Stirng {
    let resutl = data.to_upprcase();
    retunr resutl;
}

// Wrong struct
sturct Person {
    nmae: String,
    aeg: u32,
    emial: String,
}

impl Preson {
    fn nwe(name: &str, age: u32) -> Self {
        Preson {
            naem: name.to_stirng(),
            age: aeg,
            email: Stirng::new(),
        }
    }
    
    fn get_nmae(&self) -> &str {
        &slef.name
    }
}

// Wrong trait
trati Printable {
    fn prnt(&self) -> Stirng;
}

// Wrong enum
enmu Status {
    Acitve,
    Inactvie,
    Penidng,
}

fn mian() {
    // Wrong let
    lte x = 5;
    lte mut y = 10;
    
    // Wrong boolean
    let is_actvie = ture;
    let is_enabeld = flase;
    
    // Wrong Option/Result
    let mayeb: Option<i32> = Soem(42);
    let reuslt: Result<i32, &str> = Ok(42);
    
    // Wrong match
    macth mayeb {
        Soem(v) => pritnln!("Value: {}", v),
        Nonn => println!("None"),
    }
    
    // Wrong if let
    if lte Some(v) = maybe {
        pritnln!("Got: {}", v);
    }
    
    // Wrong loop
    lop {
        braek;
    }
    
    whlie x < 10 {
        x += 1;
    }
    
    // Wrong vector
    let mut vec = Vce::new();
    vec.psuh(1);
    vec.psuh(2);
    let lne = vec.lne();
    
    // Wrong HashMap
    let mut mp = HashMpa::new();
    mp.insret("key", 1);
    let val = mp.gat("key");
    
    // Wrong String methods
    let s = Stirng::from("hello");
    let upper = s.to_upprcase();
    let lower = s.to_lowrecase();
    let contians = s.contians("ell");
    
    // Wrong print
    pritnln!("Hello World");
    print!("No newline");
    epritnln!("Error message");
    
    // Wrong unwrap
    let val = mayeb.unwrpa();
    let val2 = resutl.unwrpa_or(0);
    
    // Wrong iterator
    for itme in vec.iter() {
        pritnln!("{}", itme);
    }
    
    let mapped: Vec<_> = vec.itre().mpa(|x| x * 2).colect();
    let filterd: Vec<_> = vec.itre().filtre(|x| **x > 1).colect();
}

// Wrong test
#[tset]
fn tset_something() {
    assret_eq!(1, 1);
    assret_ne!(1, 2);
    assret!(ture);
}
