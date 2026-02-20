-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.
LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY classifier IS
    GENERIC (
        d       : INTEGER := 1000;  -- Dimension size + zero-padding
        c       : INTEGER := 10;    -- Number of classes
        n       : INTEGER := 7;     -- Bit-widths of memory pointer, counter, etc.
        adI     : INTEGER := 5;     -- Number of confComp modules or adder inputs (ceiling(D / 2^n))
        adz     : INTEGER := 3;     -- Zero-padding for RSA = 2**? - adI
        zComp   : INTEGER := 6;     -- Zero-padding for Mux Comp = 2**? - c
        lgCn    : INTEGER := 4;     -- Ceiling log2(number of classes)
        logn    : INTEGER := 3      -- MuxCell RSA, ceiling log2(popCounters)
    );
    PORT (
        clk                : IN  STD_LOGIC;
        rst                : IN  STD_LOGIC;
        run                : IN  STD_LOGIC;
        hv                 : IN  STD_LOGIC_VECTOR(d - 1 DOWNTO 0);
        updated_truth      : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
        updated_prediction : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
        ground_truth       : IN  INTEGER;
        update_valid       : IN  STD_LOGIC;
        done               : OUT STD_LOGIC;
        TLAST_S            : OUT STD_LOGIC;
        TVALID_S           : OUT STD_LOGIC;
        pointer            : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0);
        classIndex         : OUT STD_LOGIC_VECTOR(lgCn-1 DOWNTO 0);
        predictedClassScore : OUT STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
        groundTruthScore    : OUT STD_LOGIC_VECTOR((n + logn) - 1 DOWNTO 0);
        update_done         : OUT STD_LOGIC
    );
END ENTITY classifier;

ARCHITECTURE behavioral OF classifier IS

    -- Counting Simulation Top-Level Component
    COMPONENT countingSimTop IS
        GENERIC (
            n              : INTEGER := 10;
            d              : INTEGER := 10;
            z              : INTEGER := 0;
            classNumber    : INTEGER := 10;
            logInNum       : INTEGER := 3;
            dimensionSize  : INTEGER := 1000
        );
        PORT (
            clk                : IN  STD_LOGIC;
            rst                : IN  STD_LOGIC;
            run                : IN  STD_LOGIC;
            hv                 : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
            update_valid       : IN  STD_LOGIC;
            updated_truth      : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
            updated_prediction : IN  STD_LOGIC_VECTOR(999 DOWNTO 0);
            ground_truth       : IN  INTEGER;
            predicted_label    : IN  INTEGER;
            done               : OUT STD_LOGIC;
            pointer            : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0);
            dout               : OUT STD_LOGIC_VECTOR(classNumber*(n+logInNum)-1 DOWNTO 0);
            update_done        : OUT STD_LOGIC
        );
    END COMPONENT;

    -- Comparator Top-Level Component
    COMPONENT comparatorTop IS
        GENERIC (
            len  : INTEGER := 8;
            n    : INTEGER := 10;
            z    : INTEGER := 10;
            lgn  : INTEGER := 4
        );
        PORT (
            clk                 : IN  STD_LOGIC;
            rst                 : IN  STD_LOGIC;
            run                 : IN  STD_LOGIC;
            a                   : IN  STD_LOGIC_VECTOR(n * len - 1 DOWNTO 0);
            ground_truth        : IN  INTEGER;
            done                : OUT STD_LOGIC;
            TLAST_S             : OUT STD_LOGIC;
            TVALID_S            : OUT STD_LOGIC;
            classIndex          : OUT STD_LOGIC_VECTOR(lgn - 1 DOWNTO 0);
            predictedClassScore : OUT STD_LOGIC_VECTOR(len - 1 DOWNTO 0);
            groundTruthScore    : OUT STD_LOGIC_VECTOR(len - 1 DOWNTO 0)
        );
    END COMPONENT;

    -- Internal Signals
    SIGNAL hvTOcount : STD_LOGIC_VECTOR(adI - 1 DOWNTO 0);
    SIGNAL dones     : STD_LOGIC;
    SIGNAL toComp    : STD_LOGIC_VECTOR(c * (n + logn) - 1 DOWNTO 0);
    SIGNAL point     : STD_LOGIC_VECTOR(n-1 DOWNTO 0);
    SIGNAL classIndexI : STD_LOGIC_VECTOR(lgCn - 1 DOWNTO 0);

BEGIN

    classIndex <= classIndexI;

    -- Generate hvTOcount using pointer indexing
    concat: FOR I IN adI-1 DOWNTO 0 GENERATE
        hvTOcount(I) <= hv(to_integer(unsigned(point)) + (2**n) * I);
    END GENERATE concat;

    -- Counting Simulation Instance
    CST : countingSimTop
        GENERIC MAP (
            n           => n,
            d           => adI,
            z           => adz,
            classNumber => c,
            logInNum    => logn
        )
        PORT MAP (
            clk                => clk,
            rst                => rst,
            run                => run,
            hv                 => hvTOcount,
            update_valid       => update_valid,
            updated_truth      => updated_truth,
            updated_prediction => updated_prediction,
            ground_truth       => ground_truth,
            predicted_label    => TO_INTEGER(unsigned(classIndexI)),
            done               => dones,
            pointer            => point,
            dout               => toComp,
            update_done        => update_done
        );

    -- Comparator Instance
    CT : comparatorTop
        GENERIC MAP (
            len  => (n + logn),
            n    => c,
            z    => zComp,
            lgn  => lgCn
        )
        PORT MAP (
            clk                 => clk,
            rst                 => rst,
            run                 => dones,
            a                   => toComp,
            ground_truth        => ground_truth,
            done                => done,
            TLAST_S             => TLAST_S,
            TVALID_S            => TVALID_S,
            classIndex          => classIndexI,
            predictedClassScore => predictedClassScore,
            groundTruthScore    => groundTruthScore
        );

    -- Assign output pointer
    pointer <= point;

END ARCHITECTURE behavioral;

