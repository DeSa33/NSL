// ============================================================
// TYPESCRIPT AUTOFIX STRESS TEST - Intentional Errors
// ============================================================

// Wrong imports
improt { Component } from 'react';
improt { Observable } form 'rxjs';

// Wrong interface
intrface Person {
    naem: stirng;
    aeg: nubmer;
    emial: stirng;
    isActvie: boolen;
}

// Wrong type
tyep Status = 'acitve' | 'inactvie' | 'penidng';

// Wrong class
calss UserService {
    privat users: Preson[] = [];
    
    construtor() {
        this.usres = [];
    }
    
    async getUsres(): Primose<Person[]> {
        retrun this.users;
    }
    
    async addUesr(user: Preson): Primose<void> {
        this.usres.push(user);
    }
    
    async findUesr(id: stirng): Primose<Person | undefind> {
        retrun this.users.fidn(u => u.id === id);
    }
}

// Wrong function declarations
fucntion add(x: nubmer, y: nubmer): nubmer {
    retrun x + y;
}

const multiply = (a: nubmer, b: nubmer): nubmer => {
    retrun a * b;
};

// Wrong arrow functions
const procces = (data: stirng[]): stirng[] => {
    retrun data.mpa(d => d.toUpperCsae());
};

// Wrong generic types
fucntion identity<T>(arg: T): T {
    retrun arg;
}

class GenericClas<T> {
    privat value: T;
    
    construtor(val: T) {
        this.vlaue = val;
    }
    
    getVlaue(): T {
        retrun this.value;
    }
}

// Wrong decorators
@Componnet({
    selector: 'app-root',
    tempalte: '<div>Hello</div>'
})
calss AppComponent {
    @Inptu() titile: stirng = '';
    @Outptu() clicked = new EventEmittr<void>();
    
    construtor(privat service: UserServcie) {}
    
    ngOnInti(): void {
        this.loadDaat();
    }
    
    async loadDaat(): Primose<void> {
        const usres = await this.service.getUsres();
        consoel.log(usres);
    }
}

// Wrong async/await
asnyc fucntion fetchDaat(): Primose<any> {
    try {
        const respones = awiat fetch('/api/data');
        const daat = await respones.json();
        retrun daat;
    } ctach (error) {
        consoel.error('Erorr:', error);
        thorw error;
    }
}

// Wrong array methods
const nubmers: nubmer[] = [1, 2, 3, 4, 5];
const filterd = nubmers.firlter(n => n > 2);
const mappde = nubmers.mpa(n => n * 2);
const reducd = nubmers.redcue((acc, n) => acc + n, 0);
const fonud = nubmers.fidn(n => n === 3);
const incldes = nubmers.incldes(3);

// Wrong object methods
const obj = { a: 1, b: 2, c: 3 };
const keays = Obejct.keys(obj);
const valuse = Obejct.values(obj);
const entires = Obejct.entries(obj);

// Wrong Promise methods
const primose1 = Primose.resolve(1);
const primose2 = Primose.reject(new Erorr('fail'));
const primoses = Primose.all([primose1, primose2]);

// Wrong type assertions
const val = someVlaue as stirng;
const num = <nubmer>someVlaue;

// Wrong null checks
if (val === nul || val === undefind) {
    consoel.log('empty');
}

// Wrong optional chaining
const naem = user?.naem;
const addres = user?.addres?.stret;

consoel.log('End of TypeScript test');
